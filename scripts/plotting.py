import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils.post_processing import smooth_sequences

def generate_all_paper_plots(dataset_name, y_true_12d, y_pred_12d, y_pred_opt, optimal_indices, flat_true_12d, flat_pred_12d, flat_true_opt, flat_pred_opt, output_dir, model_hz=50, smooth_plots=False, smooth_window=7):
    sns.set_theme(style="ticks")
    if len(y_true_12d) == 0 or len(y_pred_12d) == 0 or len(y_pred_opt) == 0 or len(optimal_indices) == 0:
        return

    optimal_indices = [int(idx) for idx in optimal_indices]
    seq_len = len(y_true_12d[0])
    if seq_len < 5: return

    frame_start = max(0, min(int(model_hz), seq_len - max(10, int(model_hz * 1.5)) - 1))
    frame_start = 0 if frame_start >= seq_len else frame_start
    short_frames = min(max(10, int(model_hz * 1.0)), max(1, seq_len - frame_start))
    long_frames = min(max(10, int(model_hz * 1.5)), max(1, seq_len - frame_start))

    time_axis = np.arange(frame_start, frame_start + short_frames) / float(model_hz)
    time_axis_long = np.arange(frame_start, frame_start + long_frames) / float(model_hz)

    true12_0, pred12_0, predopt_0 = np.asarray(y_true_12d[0], dtype=np.float32), np.asarray(y_pred_12d[0], dtype=np.float32), np.asarray(y_pred_opt[0], dtype=np.float32)

    if smooth_plots:
        pred12_0 = smooth_sequences([pred12_0], window=smooth_window)[0]
        predopt_0 = smooth_sequences([predopt_0], window=smooth_window)[0]

    top_channel = int(optimal_indices[0]) if int(optimal_indices[0]) < true12_0.shape[1] else 0

    # 1. Zoomed trajectory
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(time_axis, true12_0[frame_start:frame_start + short_frames, top_channel], label="Ground Truth", color="#2c3e50", linewidth=2.5, zorder=3)
    ax.plot(time_axis, pred12_0[frame_start:frame_start + short_frames, top_channel], label="Full 12D Baseline", color="#2980b9", linestyle="--", linewidth=2.0)
    ax.plot(time_axis, predopt_0[frame_start:frame_start + short_frames, 0], label="Optimal Subset", color="#e74c3c", linestyle=":", linewidth=2.5)
    ax.set_title(f"{dataset_name.upper()} Kinematic Tracking (Ch {top_channel})", fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.set_ylabel("Displacement (mm)", fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.grid(True, linestyle="--", alpha=0.5)
    sns.despine()
    plt.savefig(f"{output_dir}/{dataset_name}_1_Zoomed_Trajectory.png", bbox_inches="tight")
    plt.close()

    # 2. Multi-channel facet grid
    fig, axes = plt.subplots(len(optimal_indices), 1, figsize=(11, 2.2 * len(optimal_indices)), sharex=True, dpi=300, squeeze=False)
    axes = axes[:, 0]
    for idx, ch_idx in enumerate(optimal_indices):
        if ch_idx >= true12_0.shape[1] or idx >= predopt_0.shape[1]: continue
        axes[idx].plot(time_axis_long, true12_0[frame_start:frame_start + long_frames, ch_idx], color="black", linewidth=2, label="Ground Truth" if idx == 0 else "")
        axes[idx].plot(time_axis_long, predopt_0[frame_start:frame_start + long_frames, idx], color="#e74c3c", linestyle="--", linewidth=2, label="5D Subset" if idx == 0 else "")
        axes[idx].set_ylabel(f"Ch {ch_idx} (mm)", fontweight="bold")
        if idx == 0: axes[idx].legend(loc="upper right")
    axes[-1].set_xlabel("Time (seconds)", fontweight="bold")
    fig.suptitle(f"{dataset_name.upper()}: Multi-Channel Performance", fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_2_MultiChannel.png", bbox_inches="tight")
    plt.close()

    # 3. Error CDF
    valid_opt_indices = [idx for idx in optimal_indices if idx < flat_true_12d.shape[1]]
    if len(valid_opt_indices) > 0:
        error_12d = np.abs(flat_true_12d[:, valid_opt_indices] - flat_pred_12d[:, valid_opt_indices]).flatten()
        error_opt = np.abs(flat_true_opt - flat_pred_opt).flatten()
        all_errors = np.concatenate([error_12d, error_opt]) if len(error_12d) > 0 and len(error_opt) > 0 else np.array([0.0, 1.0])
        max_err = max(1.0, min(5.0, np.percentile(all_errors, 99)))
        eval_points = np.linspace(0, max_err, 200)

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.plot(eval_points, [np.mean(error_12d <= e) * 100 for e in eval_points], label="Full 12D on selected channels", color="#3498db", linewidth=2.5)
        ax.plot(eval_points, [np.mean(error_opt <= e) * 100 for e in eval_points], label="Optimal 5D", color="#e74c3c", linestyle="--", linewidth=2.5)
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{dataset_name.upper()}: Error Cumulative Distribution", fontweight="bold")
        ax.set_xlabel("Absolute Error (mm)", fontweight="bold")
        ax.set_ylabel("Frames (%)", fontweight="bold")
        ax.legend()
        sns.despine()
        plt.savefig(f"{output_dir}/{dataset_name}_3_Error_CDF.png", bbox_inches="tight")
        plt.close()

    # 4. Scatter density
    x, y = flat_true_opt.flatten(), flat_pred_opt.flatten()
    if len(x) > 100000:
        sample_idx = np.random.choice(len(x), size=100000, replace=False)
        x, y = x[sample_idx], y[sample_idx]

    if len(x) > 0:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1, bins="log")
        fig.colorbar(hb, ax=ax, label="Log10(Frame Count)")
        min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal (y = x)")
        ax.set_title(f"{dataset_name.upper()} (Optimal): Scatter Density", fontweight="bold")
        ax.set_xlabel("Ground Truth (mm)", fontweight="bold")
        ax.set_ylabel("Predicted (mm)", fontweight="bold")
        ax.legend(loc="upper left")
        sns.despine()
        plt.savefig(f"{output_dir}/{dataset_name}_4_Scatter_Density.png", bbox_inches="tight")
        plt.close()

    # 5. Inter-channel correlation heatmap
    heatmap_data = flat_true_12d
    if len(heatmap_data) > 200000:
        heatmap_data = heatmap_data[np.random.choice(len(heatmap_data), size=200000, replace=False)]

    if len(heatmap_data) > 0:
        corr = pd.DataFrame(heatmap_data, columns=[f"Ch {i}" for i in range(heatmap_data.shape[1])]).corr()
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap=sns.diverging_palette(230, 20, as_cmap=True), vmax=1.0, vmin=-1.0, center=0, square=True, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax)
        ax.set_title(f"{dataset_name.upper()}: Inter-Channel Correlation", fontweight="bold")
        plt.savefig(f"{output_dir}/{dataset_name}_5_Heatmap.png", bbox_inches="tight")
        plt.close()