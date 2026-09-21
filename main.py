import os
import gc
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from data.extraction import extract_wavlm_features
from data.dataset import prepare_dataloaders
from models.architectures import A2A_TCN_v2
from scripts.train import train_model
from scripts.evaluate import evaluate_loader
from scripts.plotting import generate_all_paper_plots
from utils.signal_processing import prepare_ema_sequences, trim_to_match, audit_dataset
from utils.post_processing import estimate_lags_by_validation, apply_lags_to_sequences
from utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "paper_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42

DATASET_CONFIG = {
    "mocha": {"audio_orig_sr": 16000, "ema_orig_hz": 500, "model_hz": 50, "trim_to_match": False},
    "usc": {"audio_orig_sr": 22050, "ema_orig_hz": 100, "model_hz": 50, "trim_to_match": False},
    "mngu0": {"audio_orig_sr": 16000, "ema_orig_hz": 200, "model_hz": 50, "trim_to_match": False},
}

summary_table = []

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_dataset(keyword, epochs=80, batch_size=32, top_k=5, do_lag=True, do_smooth=True, hidden_dim=256, dilations=(1, 2, 4, 8, 16, 32)):
    target_files = [f for f in glob.glob("**/*12D.npz", recursive=True) if keyword in os.path.basename(f).lower()]
    if not target_files:
        print(f"Dataset file not found for keyword: {keyword}")
        return None

    dataset_path = target_files[0]
    dataset_name = os.path.basename(dataset_path).replace("_baseline_12D.npz", "").replace(".npz", "").upper()
    cfg = DATASET_CONFIG.get(keyword.lower(), {})
    model_hz = cfg.get("model_hz", 50)

    base_data = np.load(dataset_path, allow_pickle=True)
    audio_arrays, ema_feats = base_data["audio"], base_data["features"]
    del base_data
    gc.collect()

    wavlm_feats = extract_wavlm_features(audio_arrays, device, orig_sr=cfg.get("audio_orig_sr", 16000), target_sr=16000)
    del audio_arrays
    gc.collect()

    wavlm_feats = [np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) for seq in wavlm_feats]
    ema_feats = prepare_ema_sequences(ema_feats, orig_hz=cfg.get("ema_orig_hz", 100), target_hz=model_hz)

    if cfg.get("trim_to_match", False):
        wavlm_feats, ema_feats = trim_to_match(wavlm_feats, ema_feats)

    valid_idx = [i for i in range(len(wavlm_feats)) if len(wavlm_feats[i]) > 0 and len(ema_feats[i]) > 0]
    wavlm_feats, ema_feats = [wavlm_feats[i] for i in valid_idx], [ema_feats[i] for i in valid_idx]

    audit_dataset(wavlm_feats, ema_feats, ema_hz=model_hz, n=5)

    input_dim, output_dim = wavlm_feats[0].shape[1], ema_feats[0].shape[1]
    top_k = min(top_k, output_dim)

    # 1. Train 12D Model
    train_12d, val_12d, test_12d, scaler_12d = prepare_dataloaders(wavlm_feats, ema_feats, batch_size=batch_size, seed=SEED)
    model_12d = A2A_TCN_v2(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, dilations=dilations).to(device)
    if torch.cuda.device_count() > 1: model_12d = nn.DataParallel(model_12d)

    res12 = train_model(model_12d, train_12d, val_12d, test_12d, f"{output_dim}D_FULL", device, scaler=scaler_12d, epochs=epochs)

    # Calculate optimal Channels
    _, _, _, _, val_pcc_12d, _ = evaluate_loader(model_12d, val_12d, device, scaler_12d)
    raw_ema_stds = np.std(np.vstack(ema_feats), axis=0)
    variance_threshold = np.percentile(raw_ema_stds[raw_ema_stds > 1e-6], 25) if np.sum(raw_ema_stds > 1e-6) >= top_k else 0.0
    valid_channel_mask = raw_ema_stds >= variance_threshold
    
    masked_val_pcc = np.where(valid_channel_mask, val_pcc_12d, -1.0)
    optimal_indices = np.argsort(np.nan_to_num(masked_val_pcc, nan=-1.0))[::-1][:top_k].tolist()

    # 2. Train Optimal Subset Model
    ema_opt = [seq[:, optimal_indices] for seq in ema_feats]
    train_5d, val_5d, test_5d, scaler_5d = prepare_dataloaders(wavlm_feats, ema_opt, batch_size=batch_size, seed=SEED)
    model_opt = A2A_TCN_v2(input_dim=input_dim, output_dim=len(optimal_indices), hidden_dim=hidden_dim, dilations=dilations).to(device)
    if torch.cuda.device_count() > 1: model_opt = nn.DataParallel(model_opt)

    res5 = train_model(model_opt, train_5d, val_5d, test_5d, f"{len(optimal_indices)}D_OPTIMAL", device, scaler=scaler_5d, epochs=epochs)

    seq_pred_12d, seq_pred_5d = res12["seq_pred"], res5["seq_pred"]
    
    if do_lag:
        max_lag = max(3, int(model_hz * 0.25))
        lags_12d = estimate_lags_by_validation(res12["val_seq_true"], res12["val_seq_pred"], max_lag=max_lag)
        lags_5d = estimate_lags_by_validation(res5["val_seq_true"], res5["val_seq_pred"], max_lag=max_lag)
        if len(lags_12d) == seq_pred_12d[0].shape[1]: res12["flat_pred"] = np.vstack(apply_lags_to_sequences(seq_pred_12d, lags_12d))
        if len(lags_5d) == seq_pred_5d[0].shape[1]: res5["flat_pred"] = np.vstack(apply_lags_to_sequences(seq_pred_5d, lags_5d))

    pcc_12d, ccc_12d = compute_metrics(res12["flat_true"], res12["flat_pred"])
    pcc_5d, ccc_5d = compute_metrics(res5["flat_true"], res5["flat_pred"])

    smooth_window = max(5, int(round(model_hz * 0.12)))
    if smooth_window % 2 == 0: smooth_window += 1

    generate_all_paper_plots(dataset_name, res12["seq_true"], seq_pred_12d, seq_pred_5d, optimal_indices, 
                             res12["flat_true"], res12["flat_pred"], res5["flat_true"], res5["flat_pred"], 
                             OUTPUT_DIR, model_hz=model_hz, smooth_plots=do_smooth, smooth_window=smooth_window)

    summary_table.append({
        "dataset": dataset_name, "model_hz": model_hz, "optimal_indices": optimal_indices,
        "12d_all_mean_pcc": float(np.mean(pcc_12d)), "12d_selected_mean_pcc": float(np.mean(np.array(pcc_12d)[optimal_indices])),
        "5d_mean_pcc": float(np.mean(pcc_5d)), "12d_all_mean_ccc": float(np.mean(ccc_12d)),
        "12d_selected_mean_ccc": float(np.mean(np.array(ccc_12d)[optimal_indices])), "5d_mean_ccc": float(np.mean(ccc_5d)),
        "12d_rmse": float(np.sqrt(np.mean((res12["flat_true"] - res12["flat_pred"]) ** 2))), 
        "5d_rmse": float(np.sqrt(np.mean((res5["flat_true"] - res5["flat_pred"]) ** 2)))
    })

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    set_seed()
    run_dataset("mocha", epochs=50)
    run_dataset("usc", epochs=50)
    run_dataset("mngu0", epochs=50)
    
    summary_df = pd.DataFrame(summary_table)
    print(summary_df)