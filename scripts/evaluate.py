import torch
import numpy as np
from utils.metrics import compute_metrics

def evaluate_loader(model, loader, device, scaler=None):
    model.eval()
    seq_true, seq_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch, lengths in loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            true = y_batch.cpu().numpy()

            for i, L in enumerate(lengths):
                L = int(L)
                seq_true.append(true[i, :L, :].astype(np.float32))
                seq_pred.append(pred[i, :L, :].astype(np.float32))

    if scaler is not None:
        seq_true = [scaler.inverse_transform(seq).astype(np.float32) for seq in seq_true]
        seq_pred = [scaler.inverse_transform(seq).astype(np.float32) for seq in seq_pred]

    if len(seq_true) == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return empty, empty, [], [], np.array([]), np.array([])

    flat_true, flat_pred = np.vstack(seq_true), np.vstack(seq_pred)
    pcc, ccc = compute_metrics(flat_true, flat_pred)

    return flat_true, flat_pred, seq_true, seq_pred, pcc, ccc