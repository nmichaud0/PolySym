import torch
from typing import Union, Callable, List

# Safe versions of operations that can cause numerical issues
def safe_sqrt(x):
    return torch.sqrt(torch.abs(x) + 1e-10)

def safe_log(x):
    return torch.log(torch.abs(x) + 1e-10)

def safe_log10(x):
    return torch.log10(torch.abs(x) + 1e-10)

def safe_inv(x):
    return 1.0 / (x + torch.sign(x) * 1e-10)  # Add small value with same sign

def safe_div(a, b):
    return a / (b + torch.sign(b) * 1e-10)  # Add small value with same sign

def safe_pow(a, b):
    # Avoid negative bases with non-integer exponents
    # We take abs of base and add small epsilon
    return torch.pow(torch.abs(a) + 1e-10, b)

def safe_asin(x):
    return torch.arcsin(torch.clamp(x, -1.0 + 1e-10, 1.0 - 1e-10))

def safe_acos(x):
    return torch.arccos(torch.clamp(x, -1.0 + 1e-10, 1.0 - 1e-10))

# Simple element-wise operators - these preserve dimensions naturally
_unary_operators_map = {
    'neg': torch.neg,
    'abs': torch.abs,
    'sqrt': safe_sqrt,
    'exp': torch.exp,
    'log': safe_log,
    'log10': safe_log10,
    'inv': safe_inv,

    'sin': torch.sin,
    'cos': torch.cos,
    'tan': torch.tan,
    'asin': safe_asin,
    'acos': safe_acos,
    'atan': torch.arctan,

    'sinh': torch.sinh,
    'cosh': torch.cosh,
    'tanh': torch.tanh,

    # Only fix reduction operators to preserve dimensions when needed
    'mean': lambda x: torch.mean(x, dim=-1) if x.dim() > 1 else torch.mean(x),
    'median': lambda x: torch.median(x, dim=-1)[0] if x.dim() > 1 else torch.median(x),
    'std': lambda x: torch.std(x, dim=-1, unbiased=False) if x.dim() > 1 else torch.std(x, unbiased=False),
    'sum': lambda x: torch.sum(x, dim=-1) if x.dim() > 1 else torch.sum(x),
    'min': lambda x: torch.min(x, dim=-1)[0] if x.dim() > 1 else torch.min(x),
    'max': lambda x: torch.max(x, dim=-1)[0] if x.dim() > 1 else torch.max(x),

    'sign': torch.sign,
}

# Standard binary operators - PyTorch handles broadcasting automatically
_binary_operators_map = {
    'add': torch.add,
    'sub': torch.sub,
    'mul': torch.mul,
    'div': safe_div,
    'pow': safe_pow,
}


class UnbatchedOperators:
    def __init__(self, operators_selection: list[str]=None, select_all: bool=False):

        self.operators_selection = operators_selection

        if self.operators_selection is None:
            if select_all is False:
                raise ValueError("operators_selection must be provided unless select_all is True")
            
            else:
                self.unary_operators = _unary_operators_map.copy()
                self.binary_operators = _binary_operators_map.copy()

        else:
            self.unary_operators = {k: v for k, v in _unary_operators_map.items() if k in operators_selection}
            self.binary_operators = {k: v for k, v in _binary_operators_map.items() if k in operators_selection}
        
    def get_unary(self) -> List[Callable]:

        return list(self.unary_operators.values())
    
    def get_binary(self) -> List[Callable]:

        return list(self.binary_operators.values())