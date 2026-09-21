import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

def apply_lag_to_channel(x, lag):
    lag = int(lag)
    if lag == 0 or len(x) == 0:
        return x
    out = np.zeros_like(x)
    if lag > 0:
        if lag < len(x): out[lag:] = x[:-lag]
    else:
        if -lag < len(x): out[:lag] = x[-lag:]
    return out

def apply_lags_to_sequences(seq_pred, lags):
    corrected = []
    for seq in seq_pred:
        s = np.array(seq, copy=True)
        for ch in range(s.shape[1]):
            if ch < len(lags):
                s[:, ch] = apply_lag_to_channel(s[:, ch], lags[ch])
        corrected.append(s.astype(np.float32))
    return corrected

def estimate_lags_by_validation(true_seqs, pred_seqs, max_lag=15):
    if len(true_seqs) == 0 or len(pred_seqs) == 0:
        return []

    n_channels = true_seqs[0].shape[1]
    max_lag = int(max(1, max_lag))
    final_lags = []

    for ch in range(n_channels):
        best_lag, best_score = 0, -np.inf
        for lag in range(-max_lag, max_lag + 1):
            t_parts, p_parts = [], []
            for t_seq, p_seq in zip(true_seqs, pred_seqs):
                if len(t_seq) <= 2 * abs(lag) + 5: continue
                tt = t_seq[:, ch]
                pp = apply_lag_to_channel(p_seq[:, ch], lag)
                
                if abs(lag) > 0:
                    tt, pp = tt[abs(lag):-abs(lag)], pp[abs(lag):-abs(lag)]
                if len(tt) < 5 or len(pp) < 5 or np.std(tt) < 1e-8 or np.std(pp) < 1e-8:
                    continue
                
                t_parts.append(tt)
                p_parts.append(pp)

            if len(t_parts) == 0: continue
            tt_all = np.concatenate(t_parts)
            pp_all = np.concatenate(p_parts)
            if np.std(tt_all) < 1e-8 or np.std(pp_all) < 1e-8: continue

            score = pearsonr(tt_all, pp_all)[0]
            score = -1.0 if np.isnan(score) else score

            if score > best_score:
                best_score, best_lag = score, lag
        final_lags.append(int(best_lag))
    return final_lags

def smooth_sequences(seqs, window=11, polyorder=2):
    if window is None or window <= 1:
        return seqs
    if window % 2 == 0:
        window -= 1

    smoothed = []
    for seq in seqs:
        if len(seq) >= window:
            try:
                seq_s = savgol_filter(seq, window, polyorder, axis=0)
                smoothed.append(seq_s.astype(np.float32))
            except Exception:
                smoothed.append(seq)
        else:
            smoothed.append(seq)
    return smoothed