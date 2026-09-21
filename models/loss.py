import torch
import torch.nn as nn

class ArticulatoryLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_var=0.1, lambda_vel=0.2, lambda_ccc=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_var = lambda_var
        self.lambda_vel = lambda_vel
        self.lambda_ccc = lambda_ccc

    def _make_mask(self, pred, lengths):
        B, T, C = pred.shape
        lengths = torch.as_tensor(lengths, device=pred.device, dtype=torch.long)
        mask = torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.unsqueeze(-1).expand(B, T, C).float()

    def _l1_loss(self, pred, target, mask):
        denom = mask.sum() + 1e-8
        return (torch.abs(pred - target) * mask).sum() / denom

    def _var_loss(self, pred, target, lengths):
        lengths = torch.as_tensor(lengths, device=pred.device, dtype=torch.long)
        losses = []
        for i in range(pred.size(0)):
            L = int(lengths[i])
            if L > 1:
                losses.append(torch.mean(torch.abs(torch.var(pred[i, :L, :], dim=0) - torch.var(target[i, :L, :], dim=0))))
        return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)

    def _velocity_loss(self, pred, target, mask):
        if pred.size(1) < 2: return pred.new_tensor(0.0)
        pred_v, target_v = pred[:, 1:, :] - pred[:, :-1, :], target[:, 1:, :] - target[:, :-1, :]
        mask_v = mask[:, 1:, :] * mask[:, :-1, :]
        return (torch.abs(pred_v - target_v) * mask_v).sum() / (mask_v.sum() + 1e-8)

    def _ccc_loss(self, pred, target, mask, eps=1e-8):
        n = mask.sum(dim=(0, 1)).clamp(min=eps)
        mean_p, mean_t = (pred * mask).sum(dim=(0, 1)) / n, (target * mask).sum(dim=(0, 1)) / n
        pred_c, target_c = (pred - mean_p) * mask, (target - mean_t) * mask
        var_p, var_t = (pred_c ** 2).sum(dim=(0, 1)) / n, (target_c ** 2).sum(dim=(0, 1)) / n
        cov = (pred_c * target_c).sum(dim=(0, 1)) / n
        ccc = (2.0 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + eps)
        return (1.0 - ccc).mean()

    def forward(self, pred, target, lengths):
        mask = self._make_mask(pred, lengths)
        loss = pred.new_tensor(0.0)
        if self.lambda_l1 > 0: loss += self.lambda_l1 * self._l1_loss(pred, target, mask)
        if self.lambda_var > 0: loss += self.lambda_var * self._var_loss(pred, target, lengths)
        if self.lambda_vel > 0: loss += self.lambda_vel * self._velocity_loss(pred, target, mask)
        if self.lambda_ccc > 0: loss += self.lambda_ccc * self._ccc_loss(pred, target, mask)
        return loss