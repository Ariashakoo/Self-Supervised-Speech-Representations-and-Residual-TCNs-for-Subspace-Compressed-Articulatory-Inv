import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MemorySafeArticulatoryDataset(Dataset):
    def __init__(self, acoustic_features, articulatory_features):
        self.X = acoustic_features
        self.Y = articulatory_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_data = np.asarray(self.X[idx], dtype=np.float32)
        y_data = np.asarray(self.Y[idx], dtype=np.float32)

        if x_data.ndim == 1: x_data = x_data[:, None]
        if y_data.ndim == 1: y_data = y_data[:, None]

        T_ema = y_data.shape[0]

        if T_ema == 0:
            x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
            y_dim = y_data.shape[1] if y_data.ndim > 1 else 1
            return (torch.zeros((1, x_dim), dtype=torch.float32), torch.zeros((1, y_dim), dtype=torch.float32))

        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        if x_tensor.shape[0] == 0:
            x_tensor = torch.zeros((T_ema, x_data.shape[1]), dtype=torch.float32)
        elif x_tensor.shape[0] != T_ema:
            x_tensor = x_tensor.unsqueeze(0).transpose(1, 2)
            x_tensor = F.interpolate(x_tensor, size=T_ema, mode="linear", align_corners=False)
            x_tensor = x_tensor.squeeze(0).transpose(0, 1)

        return x_tensor, y_tensor

def pad_collate(batch):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0.0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0.0)
    return xx_pad, yy_pad, x_lens

def prepare_dataloaders(wavlm_feats, ema_feats, batch_size=16, seed=42):
    n = len(wavlm_feats)
    indices = np.random.RandomState(seed).permutation(n)
    train_end, val_end = int(0.8 * n), int(0.9 * n)
    
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    train_wav = [np.asarray(wavlm_feats[i], dtype=np.float32) for i in train_idx]
    val_wav = [np.asarray(wavlm_feats[i], dtype=np.float32) for i in val_idx]
    test_wav = [np.asarray(wavlm_feats[i], dtype=np.float32) for i in test_idx]

    train_ema = [np.asarray(ema_feats[i], dtype=np.float32) for i in train_idx]
    val_ema = [np.asarray(ema_feats[i], dtype=np.float32) for i in val_idx]
    test_ema = [np.asarray(ema_feats[i], dtype=np.float32) for i in test_idx]

    scaler_wavlm = StandardScaler()
    for seq in train_wav: scaler_wavlm.partial_fit(seq)
    train_wav = [scaler_wavlm.transform(seq).astype(np.float32) for seq in train_wav]
    val_wav = [scaler_wavlm.transform(seq).astype(np.float32) for seq in val_wav]
    test_wav = [scaler_wavlm.transform(seq).astype(np.float32) for seq in test_wav]

    scaler_ema = StandardScaler()
    for seq in train_ema: scaler_ema.partial_fit(seq)
    train_ema = [scaler_ema.transform(seq).astype(np.float32) for seq in train_ema]
    val_ema = [scaler_ema.transform(seq).astype(np.float32) for seq in val_ema]
    test_ema = [scaler_ema.transform(seq).astype(np.float32) for seq in test_ema]

    train_loader = DataLoader(MemorySafeArticulatoryDataset(train_wav, train_ema), batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(MemorySafeArticulatoryDataset(val_wav, val_ema), batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(MemorySafeArticulatoryDataset(test_wav, test_ema), batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    return train_loader, val_loader, test_loader, scaler_ema