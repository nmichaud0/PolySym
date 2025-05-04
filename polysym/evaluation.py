import torch
from torch import Tensor

def clean(x: Tensor):
    mask = torch.isnan(x)
    return x[~mask]


def rmse(y_pred: Tensor, y_true: Tensor):
    # Root-mean-square error
    return torch.sqrt(torch.mean((clean(y_pred) - clean(y_true)) ** 2)).item()


def mse(y_pred: Tensor, y_true: Tensor):
    # Mean absolute error
    return torch.mean(torch.abs(clean(y_pred) - clean(y_true))).item()


def r2(y_pred: Tensor, y_true: Tensor):
    # R squared

    # flatten in case of multi‑dimensional targets
    yt, yp = y_true, y_pred
    y_pred = clean(y_pred).reshape(-1)
    y_true = clean(y_true).reshape(-1)

    # total sum of squares
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)

    # residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # guard against zero variance
    if ss_tot.item() == 0:
        return 0.0

    return (1.0 - ss_res / ss_tot).item()