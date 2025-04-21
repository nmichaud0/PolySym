"""from polysym.torch_operators_batched import BatchedOperators
from polysym.torch_operators_unbatched import UnbatchedOperators


class Operators:
    def __new__(cls, batched: bool, *args, **kwargs):
        target = BatchedOperators if batched else UnbatchedOperators
        return target(*args, **kwargs)
"""


# torch_operators.py
from __future__ import annotations
import torch
from typing import Callable, Dict, List
import warnings

EPS = 1e-10  # global numerical buffer


# ───────────────────────── helpers ──────────────────────────
def _as_tensor(x, ref: torch.Tensor | None = None) -> torch.Tensor:
    """Ensure *x* is a tensor (match dtype/device of *ref* if given)."""
    if torch.is_tensor(x):
        return x.to(torch.float32)
    if ref is not None and torch.is_tensor(ref):
        return torch.tensor(x, dtype=torch.float32, device=ref.device)
    return torch.tensor(x)


def maybe_unsqueeze(a, b):
    """
    Bring a|b to compatible shapes for broadcasting.
    Case handled:
        a : [N]  , b : [N,T]  ➜ a → [N,1]
        b : [N]  , a : [N,T]  ➜ b → [N,1]
    Scalars are promoted to tensors first.
    """
    a, b = _as_tensor(a, b), _as_tensor(b, a)

    if a.dim() == 1 and b.dim() == 2 and a.size(0) == b.size(0):
        return a.unsqueeze(-1), b
    if b.dim() == 1 and a.dim() == 2 and a.size(0) == b.size(0):
        return a, b.unsqueeze(-1)
    return a, b


def _binary_wrapper(func: Callable[[torch.Tensor, torch.Tensor],
                                   torch.Tensor]) -> Callable:
    """Decorator: auto‑tensorise + unsqueeze before *func*."""
    def wrapped(a, b):
        a2, b2 = maybe_unsqueeze(a, b)
        return func(a2, b2)
    return wrapped


def _unary_wrapper(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
    """Decorator: auto‑tensorise before *func*."""
    def wrapped(x):
        return func(_as_tensor(x))
    return wrapped


class UnaryOp:
    __slots__ = ("fn",)
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.fn = fn
    def __call__(self, x):
        return self.fn(_as_tensor(x))

class BinaryOp:
    __slots__ = ("fn",)
    def __init__(self, fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor]):
        self.fn = fn
    def __call__(self, a, b):
        a2, b2 = _as_tensor(a, b), _as_tensor(b, a)
        # broadcast [N] vs [N,T]
        if a2.dim()==1 and b2.dim()==2 and a2.size(0)==b2.size(0):
            a2 = a2.unsqueeze(-1)
        elif b2.dim()==1 and a2.dim()==2 and b2.size(0)==a2.size(0):
            b2 = b2.unsqueeze(-1)
        return self.fn(a2, b2)

# ─────────────────────── safe element‑wise ──────────────────
def safe_sqrt(x): return torch.sqrt(torch.abs(x) + EPS)

def safe_log(x):  return torch.log(torch.abs(x) + EPS)

def safe_log10(x): return torch.log10(torch.abs(x) + EPS)

def safe_inv(x):  return 1.0 / (x + torch.sign(x)*EPS)

def safe_asin(x): return torch.arcsin(torch.clamp(x, -1+EPS, 1-EPS))

def safe_acos(x): return torch.arccos(torch.clamp(x, -1+EPS, 1-EPS))

# ─────────────────────── reductions (dim‑safe) ──────────────
def _mean(x):   x=_as_tensor(x); return x.mean(dim=-1)      if x.dim()>1 else x.mean()
def _sum(x):    x=_as_tensor(x); return x.sum(dim=-1)       if x.dim()>1 else x.sum()
def _std(x):    x=_as_tensor(x); return x.std(dim=-1, unbiased=False) if x.dim()>1 else x.std(unbiased=False)
def _median(x): x=_as_tensor(x); return x.median(dim=-1)[0] if x.dim()>1 else x.median()
def _min(x):    x=_as_tensor(x); return x.min(dim=-1)[0]    if x.dim()>1 else x.min()
def _max(x):    x=_as_tensor(x); return x.max(dim=-1)[0]    if x.dim()>1 else x.max()

# ───────────────────── binary safe variants ─────────────────
# polysym/torch_operators.py
def safe_div(a, b):
    eps = torch.full_like(b, EPS)
    denom = b + torch.where(b == 0, eps, torch.sign(b)*EPS)
    return a / denom

def safe_pow(a, b):
    base = torch.clamp(torch.abs(a)+EPS, min=EPS)   # never exactly 0
    return torch.pow(base, b)


# after: wrap every element‑wise function
_RAW_UNARY: Dict[str, Callable] = {
    'neg':   torch.neg,   'abs':  torch.abs,
    'sqrt':  safe_sqrt,   'exp':  torch.exp,
    'log':   safe_log,    'log10': safe_log10,
    'inv':   safe_inv,    'sin':  torch.sin,
    'cos':   torch.cos,   'tan':  torch.tan,
    'asin':  safe_asin,   'acos': safe_acos,
    'atan':  torch.arctan,'sinh': torch.sinh,
    'cosh':  torch.cosh,  'tanh': torch.tanh,
}

# apply the same coercion wrapper to every one:
_unary_operators_map: Dict[str, Callable] = {
    name: UnaryOp(fn) for name, fn in {
        **_RAW_UNARY,
        # reductions (already handle scalars inside)
        'mean':   _mean,   'sum':    _sum,
        'std':    _std,    'median': _median,
        'min':    _min,    'max':    _max,
        'sign':   torch.sign
    }.items()
}


_binary_operators_map: Dict[str, Callable] = {
    'add': BinaryOp(torch.add),
    'sub': BinaryOp(torch.sub),
    'mul': BinaryOp(torch.mul),
    'div': BinaryOp(safe_div),
    'pow': BinaryOp(safe_pow)
}

class Operators:
    def __init__(self, operators_selection: list[str] = None, select_all: bool = False):

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

        # Secure warning if an operator in operators_selection is not in either _unary/_binary map

        for op in operators_selection:
            if op not in _unary_operators_map and op not in _binary_operators_map:
                warnings.warn(f'Provided operator {op} not found')

    def get_unary(self):

        return list(self.unary_operators.values())

    def get_binary(self):

        return list(self.binary_operators.values())

    def get_unary_dict(self):
        return self.unary_operators

    def get_binary_dict(self):
        return self.binary_operators
