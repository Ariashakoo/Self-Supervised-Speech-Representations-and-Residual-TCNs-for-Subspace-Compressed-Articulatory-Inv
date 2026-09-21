import copy
import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.loss import ArticulatoryLoss
from scripts.evaluate import evaluate_loader
from utils.metrics import safe_mean

def train_model(model, train_loader, val_loader, test_loader, exp_name, device, scaler=None, epochs=80, lr=3e-4, weight_decay=1e-4, patience=15):
    criterion = ArticulatoryLoss(lambda_l1=1.0, lambda_var=0.1, lambda_vel=0.2, lambda_ccc=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_score, best_weights, patience_counter = -np.inf, None, 0
    print(f"     Training {exp_name}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, lengths in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch.to(device)), y_batch.to(device), lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lengths in val_loader:
                val_loss += criterion(model(x_batch.to(device)), y_batch.to(device), lengths).item()

        val_loss /= max(1, len(val_loader))
        _, _, _, _, val_pcc, val_ccc = evaluate_loader(model, val_loader, device, scaler=scaler)
        val_score = safe_mean(val_ccc)
        scheduler.step(val_loss)

        if val_score > best_score:
            best_score, patience_counter = val_score, 0
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            best_weights = copy.deepcopy(state)
        else:
            patience_counter += 1

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"     Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PCC: {safe_mean(val_pcc):.4f} | Val CCC: {safe_mean(val_ccc):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= patience:
            print(f"  -> Early stopping at epoch {epoch+1}. Restoring best weights.")
            break

    if best_weights is not None:
        if hasattr(model, "module"): model.module.load_state_dict(best_weights)
        else: model.load_state_dict(best_weights)

    _, _, val_seq_true, val_seq_pred, _, _ = evaluate_loader(model, val_loader, device, scaler=scaler)
    flat_true, flat_pred, seq_true, seq_pred, pcc, ccc = evaluate_loader(model, test_loader, device, scaler=scaler)
    rmse = float(np.sqrt(np.mean((flat_true - flat_pred) ** 2))) if len(flat_true) > 0 else float("nan")

    return {
        "rmse": rmse, "flat_true": flat_true, "flat_pred": flat_pred, 
        "seq_true": seq_true, "seq_pred": seq_pred, "pcc": pcc, "ccc": ccc, 
        "val_seq_true": val_seq_true, "val_seq_pred": val_seq_pred
    }