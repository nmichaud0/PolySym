import torch
from torch import Tensor
from typing import Callable, override
import sympy as sp
import warnings

EPS = 1e-12

EPS      = 1e-12
INT_TOL  = 1e-6

class SafeBroadcast:
    def __init__(self, fn: Callable):

        self.fn = fn

    def __call__(self, a, b):

        a2, b2 = a, b

        if isinstance(a, float) or isinstance(a, int):
            a = torch.tensor([a], dtype=torch.float32)
        elif not torch.is_tensor(a):
            a = torch.tensor(a)
        elif a.dim() == 0:
            a = torch.tensor([a], dtype=torch.float32)

        if isinstance(b, float) or isinstance(b, int):
            b = torch.tensor([b])
        elif not torch.is_tensor(b):
            b = torch.tensor(b, dtype=torch.float32)
        elif b.dim() == 0:
            b = torch.tensor([b], dtype=torch.float32)

        if a.size(0) == b.size(0) and a.dim() != b.dim():
            if a.dim() == 1 and b.dim() == 2:
                a = a.unsqueeze(1)
            elif b.dim() == 1 and a.dim() == 2:
                b = b.unsqueeze(1)
        return self.fn(a, b)

class SafeStats(SafeBroadcast):
    @override
    def __call__(self, a, b):

        a2 = torch.tensor(a, dtype=torch.float32) if not torch.is_tensor(a) else a
        b2 = torch.tensor(b, dtype=torch.float32) if not torch.is_tensor(b) else b

        if a2.dim() != b2.dim():
            return 0

        return self.fn(a2, b2)

class SafeOp:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, a):

        if isinstance(a, float) or isinstance(a, int):
            a = torch.tensor(a, dtype=torch.float32)

        return self.fn(a)

def _nonzero(x: Tensor) -> Tensor:
    """Replace values whose magnitude is below EPS with sign(x)*EPS (or +EPS if 0)."""
    return torch.where(
        x.abs() < EPS,
        torch.where(x == 0, torch.full_like(x, EPS), torch.sign(x) * EPS),
        x,
    )

def safe_div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise a / b with no division-by-zero."""
    return a / _nonzero(b)

def safe_sqrt(x: Tensor) -> Tensor:
    """√x that never sees a negative or zero input (clamp at EPS)."""
    return torch.sqrt(torch.clamp(x, min=EPS))

def safe_log(x: Tensor) -> Tensor:
    """log(|x|+ε) or log10(|x|+ε)."""
    x_safe = x.abs() + EPS
    return torch.log(x_safe)

def safe_log10(x: Tensor):
    x_safe = x.abs() + EPS
    return torch.log10(x_safe)

def safe_pow(base: Tensor, exponent: Tensor | float) -> Tensor:
    """
    x ** p without NaNs:
      • if p ≃ integer  → torch.pow(base, round(p))  (handles negatives correctly)
      • else            → sign(base) * |base|**p
    """
    base, exponent = torch.broadcast_tensors(base, torch.as_tensor(exponent, dtype=base.dtype, device=base.device))
    exp_int = exponent.round()
    is_int  = (exponent - exp_int).abs() < INT_TOL

    out_int  = torch.pow(base, exp_int)                       # exact for integer p
    out_frac = torch.sign(base) * torch.pow(base.abs() + EPS, exponent)

    return torch.where(is_int, out_int, out_frac)

def safe_tan(x: Tensor) -> Tensor:
    """tan(x) via sin / safe cos to avoid ±Inf at π/2 + kπ."""
    return torch.sin(x) / _nonzero(torch.cos(x))

def safe_inv(x: Tensor) -> Tensor:
    """1 / x with zero-denominator protection."""
    return 1.0 / _nonzero(x)

def center(x: Tensor):
    return x - x.mean(dim=-1, keepdim=True)

def cov(a: Tensor, b: Tensor):
    return (center(a) * center(b)).mean(dim=-1)

def pearson(a: Tensor, b: Tensor):
    num = cov(a, b)
    denom = safe_sqrt(safe_pow(center(a), 2).mean(dim=-1)
                               * safe_pow(center(b), 2).mean(dim=-1))

    return safe_div(num, denom)

def _rank(x: Tensor) -> Tensor:
    idx   = torch.argsort(x, dim=-1)
    rank  = torch.argsort(idx, dim=-1).to(torch.float32)
    return rank + 1.0                                  # ranks start at 1

def spearman(a: Tensor, b: Tensor) -> Tensor:
    return pearson(_rank(a), _rank(b))

def op_mean(x: Tensor) -> Tensor:
    return x.mean(dim=-1)

def op_median(x: Tensor) -> Tensor:
    return x.median(dim=-1).values

def op_sum(x: Tensor) -> Tensor:
    return x.sum(dim=-1)

def op_std(x: Tensor) -> Tensor:
    return x.std(dim=-1, unbiased=False)

def op_min(x: Tensor) -> Tensor:
    return x.min(dim=-1).values

def op_max(x: Tensor) -> Tensor:
    return x.max(dim=-1).values

# sympy printers
def sym_neg(x):        return -x
def sym_abs(x):        return sp.Abs(x)
def sym_sin(x):        return sp.sin(x)
def sym_cos(x):        return sp.cos(x)
def sym_tan(x):        return sp.tan(x)
def sym_sqrt(x):       return sp.sqrt(x)
def sym_exp(x):        return sp.exp(x)
def sym_log(x):        return sp.log(x)
def sym_log10(x):      return sp.log(x)
def sym_center(x):     return sp.Function('center')(x)

def sym_mean(x):       return sp.Function('mean')(x)
def sym_median(x):     return sp.Function('median')(x)
def sym_sum(x):        return sp.Function('sum')(x)
def sym_std(x):        return sp.Function('std')(x)
def sym_min(x):        return sp.Function('min')(x)
def sym_max(x):        return sp.Function('max')(x)

def sym_add(a, b):     return sp.Add(a, b)
def sym_sub(a, b):     return a - b
def sym_mul(a, b):     return sp.Mul(a, b)
def sym_div(a, b):     return a / b
def sym_pow(a, b):     return sp.Pow(a, b)

def sym_cov(a, b):     return sp.Function('cov')(a, b)
def sym_pearsonr(a, b):return sp.Function('pearsonr')(a, b)
def sym_spearmanr(a, b): return sp.Function('spearmanr')(a, b)

UNARY_NONREDUCE = {
    'neg': (SafeOp(torch.neg), sym_neg),
    'abs': (SafeOp(torch.abs), sp.Abs),
    'sin': (SafeOp(torch.sin), sp.sin),
    'cos': (SafeOp(torch.cos), sp.cos),
    'tan': (SafeOp(safe_tan), sp.tan),
    'sqrt': (SafeOp(safe_sqrt), sp.sqrt),
    'exp': (SafeOp(torch.exp), sp.exp),
    'log': (SafeOp(safe_log), sp.log),
    'log10': (SafeOp(safe_log10), sp.log),
    'center': (SafeOp(center), sym_center)
}

UNARY_REDUCE = {
    'mean': (SafeOp(op_mean), sym_mean),
    'median': (SafeOp(op_median), sym_median),
    'sum': (SafeOp(op_sum), sym_sum),
    'std': (SafeOp(op_std), sym_std),
    'min': (SafeOp(op_min), sym_min),
    'max': (SafeOp(op_max), sym_max)
}

BINARY_NONREDUCE = {
    'add': (SafeBroadcast(torch.add), sym_add),
    'sub': (SafeBroadcast(torch.sub), sym_sub),
    'mul': (SafeBroadcast(torch.mul), sym_mul),
    'div': (SafeBroadcast(safe_div), sym_div),
    'pow': (SafeBroadcast(safe_pow), sym_pow)
}

BINARY_REDUCE = {
    'cov': (SafeStats(cov), sym_cov),
    'pearsonr': (SafeStats(pearson), sym_pearsonr),
    'spearmanr': (SafeStats(spearman), sym_spearmanr),
}

UNARY_OPS = {k: v[0] for d in (UNARY_NONREDUCE, UNARY_REDUCE) for k, v in d.items()}
BINARY_OPS = {k: v[0] for d in (BINARY_NONREDUCE, BINARY_REDUCE) for k, v in d.items()}

class Operators:
    """
    Pick-and-choose subset of operators while preserving meta-info.
    `operators_selection` is a list like ["add","sub","mean",…].
    """
    def __init__(self, operators_selection = None, select_all: bool = False):

        if operators_selection is None and not select_all:
            raise ValueError("Provide `operators_selection` or set `select_all=True`.")

        wanted = set(operators_selection) if operators_selection else set()
        all_ops = (set(UNARY_NONREDUCE) | set(UNARY_REDUCE) | set(BINARY_NONREDUCE) | set(BINARY_REDUCE))
        missing = wanted - all_ops
        if missing:
            warnings.warn(f'Operators not found: {sorted(missing)}')

        # Filter each category
        def _filter(src: dict):
            if not wanted:
                return src.copy()
            return {k: v for k, v in src.items() if k in wanted}

        self.unary_nonreduce  = _filter(UNARY_NONREDUCE)
        self.unary_reduce     = _filter(UNARY_REDUCE)
        self.binary_nonreduce = _filter(BINARY_NONREDUCE)
        self.binary_reduce    = _filter(BINARY_REDUCE)

        # Public views expected by the rest of Polysym
        self.unary_operators  = {k: v[0] for d in (self.unary_nonreduce,
                                                   self.unary_reduce)
                                                   for k,v in d.items()}
        self.binary_operators = {k: v[0] for d in (self.binary_nonreduce,
                                                   self.binary_reduce)
                                                   for k,v in d.items()}

        self.map = self._build_sympy_map()   # for compile_tree()

    # ──────────────── helpers ──────────────────
    def _build_sympy_map(self):
        mp = {}
        for name, (_, symfn) in {**self.unary_nonreduce,
                                 **self.unary_reduce}.items():
            mp[f"unary_{name}"]  = symfn
        for name, (_, symfn) in {**self.binary_nonreduce,
                                 **self.binary_reduce}.items():
            mp[f"binary_{name}"] = symfn
        return mp

    # convenience getters (used elsewhere)
    def get_unary(self):  return list(self.unary_operators.values())
    def get_binary(self): return list(self.binary_operators.values())

    def get_unary_dict(self): return self.unary_operators
    def get_binary_dict(self): return self.binary_operators