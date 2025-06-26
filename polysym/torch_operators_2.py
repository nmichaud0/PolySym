import torch
from torch import Tensor
from typing import Callable, override
import sympy as sp
import warnings
from math import isnan

EPS = 1e-12

def _nan_reduce(x: Tensor, red_fn, *, with_values: bool = False) -> Tensor:
    """
    Reduce `x` along the last (time) axis with `red_fn` (e.g. torch.nanmean).

    *If any timestep in the slice is NaN, the reduced scalar is set to NaN*.
    This guarantees that the NaN mask of a scalar reducer matches the
    element‑wise mask expected by subsequent broadcasting checks.
    """
    mask_any_nan = torch.isnan(x).any(dim=-1)          # True if *any* NaN

    if with_values:                                    # torch.nanmedian returns namedtuple
        out = red_fn(x, dim=-1).values
    else:
        out = red_fn(x, dim=-1)

    out[mask_any_nan] = float("nan")
    return out


class SafeBroadcast:
    def __init__(self, fn: Callable, nan_fn=None):

        self.fn = fn
        self.nan_fn = fn if nan_fn is None else nan_fn

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

        if torch.isnan(a).any() or torch.isnan(b).any():
            return self.nan_fn(a, b)

        return self.fn(a, b)

class SafeStats(SafeBroadcast):
    @override
    def __call__(self, a, b):

        a2 = torch.tensor(a, dtype=torch.float32) if not torch.is_tensor(a) else a
        b2 = torch.tensor(b, dtype=torch.float32) if not torch.is_tensor(b) else b

        if a2.dim() != b2.dim():
            return 0

        return super().__call__(a, b)

        #return self.fn(a2, b2)

class SafeOp:
    def __init__(self, fn, nan_fn=None):
        self.fn = fn
        self.nan_fn = fn if nan_fn is None else nan_fn

    def __call__(self, a):

        if isinstance(a, float) or isinstance(a, int):
            a = torch.tensor(a, dtype=torch.float32)
        if torch.isnan(a).any():
            return self.nan_fn(a)

        return self.fn(a)

def _nonzero(x: Tensor):
    """Replace values whose magnitude is below EPS with sign(x)*EPS (or +EPS if 0)."""
    return torch.where(
        x.abs() < EPS,
        torch.where(x == 0, torch.full_like(x, EPS), torch.sign(x) * EPS),
        x,
    )

def safe_div(a: Tensor, b: Tensor):
    """Element-wise a / b with no division-by-zero."""
    return a / _nonzero(b)

def safe_sqrt(x: Tensor):
    """√x that never sees a negative or zero input (clamp at EPS)."""
    return torch.sqrt(torch.clamp(x, min=EPS))

def safe_square(x: Tensor):
    return x ** 2

# TODO: log don't do abs. replace all <= 0 by EPS
def safe_log(x: Tensor):
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
    is_int  = (exponent - exp_int).abs() < EPS

    out_int  = torch.pow(base, exp_int)                       # exact for integer p
    out_frac = torch.sign(base) * torch.pow(base.abs() + EPS, exponent)

    return torch.where(is_int, out_int, out_frac)

def safe_tan(x: Tensor) -> Tensor:
    """tan(x) via sin / safe cos to avoid ±Inf at π/2 + kπ."""
    return torch.sin(x) / _nonzero(torch.cos(x))

def safe_inv(x: Tensor) -> Tensor:
    """1 / x with zero-denominator protection."""
    return 1.0 / _nonzero(x)

"""def center(x: Tensor):
    mean = op_mean(x)
    return x - mean.unsqueeze(-1)
    # return x - x.nanmean(dim=-1, keepdim=True)
"""
def center(x: Tensor):
    """
    Subtract the NaN‑aware mean from each vector **while preserving the
    original NaN pattern**.  Positions that were NaN stay NaN; no new NaNs
    are introduced anywhere else.
    """

    return x - op_mean(x).unsqueeze(1)

    mean = torch.nanmean(x, dim=-1, keepdim=True)   # shape (..., 1)
    out  = x - mean
    out[torch.isnan(x)] = float("nan")              # restore NaNs only where they were
    return out

def cov(a: Tensor, b: Tensor):

    a0 = a - torch.nanmean(a, dim=-1, keepdim=True)
    b0 = b - torch.nanmean(b, dim=-1, keepdim=True)

    r = torch.nanmean(a0 * b0, dim=-1)

    return torch.nan_to_num(r, nan=float('-inf'))

    # return op_mean(center(a) * center(b))
    # return torch.nanmean((center(a) * center(b)), dim=-1)

"""def pearson(a: Tensor, b: Tensor):
    num = cov(a, b)
    denom = safe_sqrt(safe_pow(center(a), 2).mean(dim=-1)
                               * safe_pow(center(b), 2).mean(dim=-1))

    return safe_div(num, denom)
"""
# replace the whole pearson() body -------------------------------------
def pearson(a: Tensor, b: Tensor):
    """
    NaN‑propagating Pearson correlation.
    Returns NaN for each slice where either input contains a NaN.
    """
    num   = cov(a, b)                          # already NaN‑prop
    denom = op_std(a) * op_std(b)              # use NaN‑prop std
    return safe_div(num, denom)

"""def _rank(x: Tensor):
    idx   = torch.argsort(x, dim=-1)
    rank  = torch.argsort(idx, dim=-1).to(torch.float32)
    return rank + 1.0                                  # ranks start at 1
"""
def _rank(x: torch.Tensor) -> torch.Tensor:
    """
    Return ranks 1…N for non‑NaN elements along the last dim and keep NaNs
    where they were.  Equal values get average rank (stable).
    """
    shape = x.shape
    flat  = x.reshape(-1, shape[-1])          # (batch, L)
    out   = torch.full_like(flat, float("nan"))
    for i, row in enumerate(flat):
        valid = ~torch.isnan(row)
        if valid.sum() == 0:
            continue                          # keep all‑NaN row as NaN
        vals = row[valid]
        order = torch.argsort(vals)
        ranks = torch.empty_like(vals)
        ranks[order] = torch.arange(1, valid.sum() + 1, dtype=torch.float32)
        out[i, valid] = ranks
    return out.reshape(shape)

def spearman(a: Tensor, b: Tensor):

    """if torch.isnan(a).any() or torch.isnan(b).any():
        return spearman_nan(a, b)"""

    return pearson(_rank(a), _rank(b))

"""def op_mean(x: Tensor):

    return _nan_reduce(x, torch.nanmean)
    # return _propagate_nan(x, torch.nanmean)
    # return x.nanmean(dim=-1)

def op_median(x: Tensor):

    return _nan_reduce(x, torch.nanmedian, with_values=True)

    #return _propagate_nan(x, torch.nanmedian)
    # return x.nanmedian(dim=-1).values

def op_sum(x: Tensor):

    return _nan_reduce(x, torch.nansum)
    #return _propagate_nan(x, torch.nansum)
    # return x.nansum(dim=-1)

# add just after op_std ------------------------------------------------
def op_std(x: Tensor):
    mask_any_nan = torch.isnan(x).any(dim=-1)
    out = torch.std(x, dim=-1, unbiased=False)
    out[mask_any_nan] = float("nan")
    return out"""


def op_mean(x: torch.Tensor) -> torch.Tensor:
    """NaN‑aware mean over the last axis, implemented with explicit loops."""
    n_obs, T = x.shape
    out = torch.empty(n_obs, dtype=x.dtype, device=x.device)
    for i in range(n_obs):
        row = x[i]
        valid = row[~torch.isnan(row)]
        out[i] = valid.mean()
    return out


def op_median(x: torch.Tensor) -> torch.Tensor:
    """NaN‑aware median over the last axis, explicit loops."""
    n_obs, T = x.shape
    out = torch.empty(n_obs, dtype=x.dtype, device=x.device)
    for i in range(n_obs):
        row = x[i]
        valid = row[~torch.isnan(row)]
        out[i] = valid.median()

    return out


def op_sum(x: torch.Tensor) -> torch.Tensor:
    """NaN‑aware sum over the last axis, explicit loops."""
    n_obs, T = x.shape
    out = torch.empty(n_obs, dtype=x.dtype, device=x.device)
    for i in range(n_obs):
        row = x[i]
        valid = row[~torch.isnan(row)]
        out[i] = valid.sum()
    return out


def op_std(x: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    NaN‑aware standard deviation over the last axis.

    unbiased=False ⇒ population std (divide by N)
    unbiased=True  ⇒ sample   std (divide by N‑1, requires N>1)
    """
    n_obs, T = x.shape
    out = torch.empty(n_obs, dtype=x.dtype, device=x.device)
    for i in range(n_obs):
        obs = x[i, :]
        valid = obs[~torch.isnan(obs)]
        n = valid.numel()
        if n == 0 or (unbiased and n == 1):
            out[i] = float("nan")
        else:
            mean = valid.mean()
            var = ((valid - mean) ** 2).sum() / (n - 1 if unbiased else n)
            out[i] = torch.sqrt(var)
    return out

def op_min(x: Tensor):
    return torch.nan_to_num(x, nan=float('inf')).min(dim=-1).values

    #return x.min(dim=-1).values

def op_max(x: Tensor):

    return torch.nan_to_num(x, nan=float('-inf')).max(dim=-1).values

    # return x.max(dim=-1).values

# NAN SAFE:
def op_mean_nan(x: Tensor):
    return torch.nanmean(x, dim=-1)

def op_median_nan(x: Tensor):
    return torch.nanmedian(x, dim=-1).values

def op_sum_nan(x: Tensor):
    return torch.nansum(x, dim=-1)

def center_nan(x: Tensor):
    return x - torch.nanmean(x, dim=-1, keepdim=True)

def cov_nan(a: Tensor, b: Tensor):
    return op_mean_nan(center_nan(a) * center_nan(b))

def _rank_nan(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    flat   = x.reshape(-1, orig_shape[-1])          # (batch, L)
    ranks  = torch.full_like(flat, float("nan"))
    for i, row in enumerate(flat):
        # mask valid entries
        valid = ~torch.isnan(row)
        if valid.sum() == 0:
            continue
        vals = row[valid]
        # argsort twice gives rank order (0-based)
        order = torch.argsort(vals)
        ranks_i = torch.empty_like(vals)
        ranks_i[order] = torch.arange(1, valid.sum()+1, dtype=torch.float32)
        ranks[i, valid] = ranks_i
    return ranks.reshape(orig_shape)

def pearson_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_mean = torch.nanmean(a, dim=-1, keepdim=True)
    b_mean = torch.nanmean(b, dim=-1, keepdim=True)
    a_c = a - a_mean
    b_c = b - b_mean
    num   = torch.nanmean(a_c * b_c, dim=-1)
    denom = torch.std(a, dim=-1) * torch.std(b, dim=-1)
    return num / denom

def spearman_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rank_a = _rank_nan(a)
    rank_b = _rank_nan(b)
    return pearson_nan(rank_a, rank_b)

# sympy printers
def sym_square(x):     return x ** 2

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

# Third variable in tuple is required rank
# 0 means free (for binary: either s-s / v-v / s-v)
# 1 means scalar only
# 2 means vector only
# 3 means one vector one scalar

UNARY_NONREDUCE = {
    'neg': (SafeOp(torch.neg), sym_neg, 0),
    'abs': (SafeOp(torch.abs), sp.Abs, 0),
    'sin': (SafeOp(torch.sin), sp.sin, 0),
    'cos': (SafeOp(torch.cos), sp.cos, 0),
    'tan': (SafeOp(safe_tan), sp.tan, 0),
    'sqrt': (SafeOp(safe_sqrt), sp.sqrt, 0),
    'square': (SafeOp(safe_square), sym_square, 0),
    'exp': (SafeOp(torch.exp), sp.exp, 0),
    'log': (SafeOp(safe_log), sp.log, 0),
    'log10': (SafeOp(safe_log10), sp.log, 0),
    'center': (SafeOp(center), sym_center, 2)  # requires vector
}

UNARY_REDUCE = {
    'mean': (SafeOp(op_mean), sym_mean, 2),
    'median': (SafeOp(op_median), sym_median, 2),
    'sum': (SafeOp(op_sum), sym_sum, 2),
    'std': (SafeOp(op_std), sym_std, 2),
    'min': (SafeOp(op_min), sym_min, 2),
    'max': (SafeOp(op_max), sym_max, 2)
}

BINARY_NONREDUCE = {
    'add': (SafeBroadcast(torch.add), sym_add, 0),
    'sub': (SafeBroadcast(torch.sub), sym_sub, 0),
    'mul': (SafeBroadcast(torch.mul), sym_mul, 0),
    'div': (SafeBroadcast(safe_div), sym_div, 0),
    # 'pow': (SafeBroadcast(safe_pow), sym_pow, 3)  # power would break and increase search space
}

BINARY_REDUCE = {
    'cov': (SafeStats(cov), sym_cov, 2),
    'pearsonr': (SafeStats(pearson), sym_pearsonr, 2),
    'spearmanr': (SafeStats(spearman), sym_spearmanr, 2),
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

        self.map = self._build_sympy_map2()   # for compile_tree()

    # ──────────────── helpers ──────────────────
    def _build_sympy_map(self):
        mp = {}
        for name, (_, symfn, _) in {**self.unary_nonreduce,
                                 **self.unary_reduce}.items():
            mp[f"unary_{name}"]  = symfn
        for name, (_, symfn, _) in {**self.binary_nonreduce,
                                 **self.binary_reduce}.items():
            mp[f"binary_{name}"] = symfn
        return mp


    def _build_sympy_map2(self):
        mp = {}
        # unary
        for name, (_, symfn, _) in {**self.unary_nonreduce,
                                    **self.unary_reduce}.items():
            mp[name] = symfn  # ← no "unary_" prefix
        # binary
        for name, (_, symfn, _) in {**self.binary_nonreduce,
                                    **self.binary_reduce}.items():
            mp[name] = symfn  # ← no "binary_" prefix
        return mp

    # convenience getters (used elsewhere)
    def get_unary(self):  return list(self.unary_operators.values())
    def get_binary(self): return list(self.binary_operators.values())

    def get_unary_dict(self): return self.unary_operators
    def get_binary_dict(self): return self.binary_operators