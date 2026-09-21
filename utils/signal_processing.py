import math
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

def fix_audio(x, orig_sr=None, target_sr=16000):
    x = np.asarray(x)
    if x.ndim > 1:
        if x.shape[0] <= 8 and x.shape[0] < x.shape[1]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x

    x = x - np.mean(x)
    peak = np.abs(x).max()
    if peak > 1e-8:
        if peak > 1000:
            x = x / 32768.0
        else:
            x = x / peak
        x = x * 0.95

    if orig_sr is not None and int(orig_sr) != int(target_sr):
        g = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        x = resample_poly(x, up, down).astype(np.float32)

    return x

def clean_ema(seq):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq[:, None]

    if np.isnan(seq).any():
        df = pd.DataFrame(seq)
        df = df.interpolate(axis=0, limit_direction="both")
        df = df.ffill().bfill()
        seq = df.to_numpy(dtype=np.float32)

    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    return seq

def resample_ema(seq, orig_hz, target_hz):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq[:, None]

    if orig_hz is None or target_hz is None or int(orig_hz) == int(target_hz):
        return seq

    if len(seq) < 8:
        return seq

    g = math.gcd(int(orig_hz), int(target_hz))
    up = int(target_hz) // g
    down = int(orig_hz) // g

    seq = resample_poly(seq, up, down, axis=0).astype(np.float32)
    return seq

def prepare_ema_sequences(ema_feats, orig_hz, target_hz):
    prepared = []
    for seq in ema_feats:
        seq = clean_ema(seq)
        seq = resample_ema(seq, orig_hz=orig_hz, target_hz=target_hz)
        prepared.append(seq.astype(np.float32))
    return prepared

def trim_to_match(wavlm_feats, ema_feats):
    trimmed_wav, trimmed_ema = [], []
    for w, e in zip(wavlm_feats, ema_feats):
        L = min(len(w), len(e))
        if L > 0:
            trimmed_wav.append(w[:L].astype(np.float32))
            trimmed_ema.append(e[:L].astype(np.float32))
    return trimmed_wav, trimmed_ema

def audit_dataset(wavlm_feats, ema_feats, ema_hz=50, n=5, wavlm_fps=50):
    print("\nAUDIT: Frame-rate consistency")
    print("ratio = EMA frames / WavLM frames after EMA resampling")
    n = min(n, len(wavlm_feats), len(ema_feats))
    for i in range(n):
        wavlm_dur = len(wavlm_feats[i]) / float(wavlm_fps)
        ema_dur = len(ema_feats[i]) / float(ema_hz)
        ratio = len(ema_feats[i]) / max(1, len(wavlm_feats[i]))
        expected = float(ema_hz) / float(wavlm_fps)
        print(f" Sample {i}: wavlm={wavlm_dur:.2f}s | ema={ema_dur:.2f}s | ratio={ratio:.3f} | expected={expected:.3f}")
    print("-" * 70)