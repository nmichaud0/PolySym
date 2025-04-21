import torch
from torch import Tensor


def rmse(y_pred: Tensor, y_true: Tensor):
    # Root-mean-square error
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def mse(y_pred: Tensor, y_true: Tensor):
    # Mean absolute error
    return torch.mean(torch.abs(y_pred - y_true)).item()
