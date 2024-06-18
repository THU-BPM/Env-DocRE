import torch
import torch.nn as nn


class ClsLoss(nn.Module):
    def __init__(self, shift=0.0, regularization=True, eps=1e-8):
        super().__init__()
        self.shift = shift
        self.regularization = regularization
        self.eps = eps

    def ASL_calculate_CE(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()

    def forward(self, logits, labels):
        rel_margin = logits[:, 1:] - logits[:, :1]
        loss = self.ASL_calculate_CE(rel_margin.float(), labels[:, 1:].float())

        if self.regularization:
            na_margin = logits[:, 0] - logits[:, 1:].mean(-1)
            loss += self.ASL_calculate_CE(na_margin.float(), labels[:, 0].float())

        loss /= labels.shape[0]
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class PcrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, base_logits, crt_logits):
        base_logits = (base_logits[:, 1:] - base_logits[:, :1]).float()
        crt_logits = (crt_logits[:, 1:] - crt_logits[:, :1]).float()
        loss_1 = self.bce_logits(crt_logits, torch.sigmoid(base_logits))
        loss_2 = self.bce_logits(base_logits, torch.sigmoid(crt_logits))
        return (loss_1 + loss_2) / 2 / base_logits.shape[0]


class RcrLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, base_bl, crt_bl):
        return self.mse(base_bl, crt_bl)
