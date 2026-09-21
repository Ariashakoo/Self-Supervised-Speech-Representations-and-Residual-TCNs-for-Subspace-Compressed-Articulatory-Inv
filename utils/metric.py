import numpy as np
from scipy.stats import pearsonr

def safe_mean(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return -1.0 if len(arr) == 0 else float(np.mean(arr))

def concordance_correlation_coefficient(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2 or np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0

    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))

    ccc = (2.0 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-8)
    return 0.0 if np.isnan(ccc) else float(ccc)

def compute_metrics(flat_true, flat_pred):
    pcc_values, ccc_values = [], []
    if flat_true.shape[1] != flat_pred.shape[1]:
        raise ValueError("Inputs must have the same number of channels.")

    for c in range(flat_true.shape[1]):
        t, p = flat_true[:, c], flat_pred[:, c]
        if len(t) < 2 or np.std(t) < 1e-8 or np.std(p) < 1e-8:
            pcc_values.append(0.0)
            ccc_values.append(0.0)
            continue

        r = pearsonr(t, p)[0]
        pcc_values.append(0.0 if np.isnan(r) else float(r))
        ccc_values.append(concordance_correlation_coefficient(t, p))

    return np.array(pcc_values), np.array(ccc_values)