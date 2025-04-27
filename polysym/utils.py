import logging
import random
import numpy as np
import torch
from functools import cache
from typing import Callable
import copy
from deap import gp
from torch import Tensor
import sympy as sp
import re

class _RandConst:
    """Pickleable constant‐generator with per‐instance bounds."""
    def __init__(self, min_c: float, max_c: float):
        self.min_c = min_c
        self.max_c = max_c
    def __call__(self) -> float:
        return round(random.uniform(self.min_c, self.max_c), 3)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger(name: str = "polysym", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def _round_floats(expr: sp.Expr, ndigits: int = 2) -> sp.Expr:
    """
    Walk the expression and replace every Float literal
    with a rounded version to `ndigits` decimals.
    """
    # collect all Float atoms
    old_floats = list(expr.atoms(sp.Float))
    # build a mapping old→new
    repl = {
        f: sp.Float(round(float(f), ndigits), ndigits)
        for f in old_floats
    }
    # do the replacement in one go
    return expr.xreplace(repl)

@cache
def compile_tree(expr_str: str, pset) -> Callable:
    """
    Compile a GP tree *once* for each unique string representation.
    """
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    return gp.compile(expr=tree, pset=pset)


def _hill_climb_constants(ind, inputs, pset, fitness_fn, y,
                          objective, worst_fitness,
                          steps: int = 100,
                          sigma: float = .25):
    """
    Very simple hill‑climbing over the ephemeral 'randc' constants in `ind`.
    Returns a list of new constant values of the same length/order.
    """
    # find where the ephemerals live
    eph = _extract_ephemerals(ind)
    if not eph:
        return []
    idxs, orig_vals = zip(*eph)
    best_vals = list(orig_vals)
    best_fit = worst_fitness
    best_ind = ind

    # evaluate a fresh copy with given vals
    def eval_with(vals):
        tmp = copy.deepcopy(ind)
        tmp = _apply_ephemerals(tmp, vals, eph)
        # func = gp.compile(expr=tmp, pset=pset)
        func = compile_tree(str(tmp), pset)
        out = func(*inputs)
        if out.dim() != objective:
            return worst_fitness, tmp
        return float(fitness_fn(out, y)), tmp

    # initialize
    fit0, tmp_ind = eval_with(best_vals)
    if fit0 < best_fit:
        best_fit = fit0
        best_ind = tmp_ind

    # simple gaussian perturbations
    for _ in range(steps):
        cand = [v + sigma * random.gauss(0,1) for v in best_vals]
        f, ind_tmp = eval_with(cand)
        if f < best_fit:
            best_fit, best_vals, best_ind = f, cand, ind_tmp

    return best_vals, best_fit, best_ind


def _extract_ephemerals(ind: gp.PrimitiveTree):
    """
    Find all the ephemeral‐constant nodes (name 'randc') in the tree
    and return their (index, original_value) in tree order.
    """
    out = []
    for idx, node in enumerate(ind):
        # ephemeral constants are stored as gp.Terminal with name 'randc'
        if isinstance(node, gp.Terminal) and node.name == 'randc':
            out.append((idx, node.value))
    return out

def _apply_ephemerals(ind: gp.PrimitiveTree, new_values: list[float], extracted_ephemerals=None):
    """
    Overwrite only the actual ephemeral‐constant nodes ('randc')
    in the individual with the new_values.
    """
    if extracted_ephemerals is None:
        eph = _extract_ephemerals(ind)
    else:
        eph = extracted_ephemerals

    if len(eph) != len(new_values):
        raise ValueError(f"Expected {len(eph)} ephemerals, got {len(new_values)}")

    for ((idx, old_val), new_val) in zip(eph, new_values):
        node = ind[idx]
        # sanity check
        assert isinstance(node, gp.Terminal) and hasattr(node, "value"), f'Node as position {idx} is not an ephemeral constant'

        node.value = new_val

    return ind

def _evaluate_worker(ind,
                     inputs,
                     hill_inputs,
                     pset,
                     fitness_fn,
                     y,
                     y_hill,
                     worst_fitness,
                     objective,
                     optimize_ephemerals,
                     fitness_obj,
                     threshold,
                     opt_steps,
                     opt_sigma,
                     ngsa_alpha):
    """
    Multiprocessing implementation

    returns fitness, dimensionality mismatch (whether tree doesn't return expected shape),
    and individual
    flag tells if ephemeral optimization was done

    """
    # func = gp.compile(expr=ind, pset=pset)
    func = compile_tree(str(ind), pset)

    raw = func(*inputs)

    dimensionality_mismatch = raw.dim() != objective

    # Do we have ephemerals ?
    eph = bool([v for _, v in _extract_ephemerals(ind)])

    fitness = worst_fitness if dimensionality_mismatch else fitness_fn(raw, y)

    if fitness_obj == -1:
        do_opt = fitness <= threshold
    else:
        do_opt = fitness > threshold

    if not eph or not optimize_ephemerals or not do_opt or dimensionality_mismatch:
        if fitness != worst_fitness:

            if fitness_obj == -1:
                fitness += ngsa_alpha * ind.height
            else:
                fitness -= ngsa_alpha * ind.height

        return fitness, dimensionality_mismatch, False, ind

    ephemerals, better_fit, better_ind = _hill_climb_constants_ls(ind, hill_inputs, pset, fitness_fn, y_hill, objective, worst_fitness, steps=opt_steps, sigma=opt_sigma)

    if fitness_obj == -1:
        # if opt failed
        if better_fit > fitness:
            better_fit = fitness
            better_ind = ind
        better_fit += ngsa_alpha * better_ind.height #better_ind.height
    else:
        if better_fit < fitness:
            better_fit = fitness
            better_ind = ind
        better_fit -= ngsa_alpha * better_ind.height

    return better_fit, dimensionality_mismatch, True, better_ind


def _hill_climb_constants_ls(ind, inputs, pset, fitness_fn, y,
                             objective, worst_fitness,
                             steps: int = 1,
                             sigma: float = 1e-3):
    """
    Gauss–Newton least-squares update of ephemeral constants.

    Assumes small perturbations; builds a Jacobian by finite differences
    and solves (JᵀJ + λI) Δc = Jᵀ (y - f(c)) for the update Δc.
    """
    eph = _extract_ephemerals(ind)
    if not eph:
        return [], worst_fitness, ind

    idxs, orig_vals = zip(*eph)
    best_vals = np.array(orig_vals, dtype=float)
    # Evaluate original output
    tmp = copy.deepcopy(ind)
    tmp = _apply_ephemerals(tmp, best_vals.tolist(), eph)
    func = compile_tree(str(tmp), pset)
    out0 = func(*inputs)
    if out0.dim() != objective:
        return best_vals.tolist(), worst_fitness, ind

    # Flatten outputs
    out0_flat = out0.detach().cpu().numpy().reshape(-1)
    y_flat    = y.detach().cpu().numpy().reshape(-1)
    residual  = y_flat - out0_flat

    # Build Jacobian J_{ij} = ∂f_i/∂c_j ≈ [f(c+ε e_j) - f(c)]/ε
    n = out0_flat.size
    m = best_vals.size
    J = np.zeros((n, m), dtype=float)
    eps = sigma
    for j in range(m):
        vals = best_vals.copy()
        vals[j] += eps
        tmpj = copy.deepcopy(ind)
        tmpj = _apply_ephemerals(tmpj, vals.tolist(), eph)
        funcj = compile_tree(str(tmpj), pset)
        outj = funcj(*inputs).detach().cpu().numpy().reshape(-1)
        J[:, j] = (outj - out0_flat) / eps

    # Solve normal equations: (JᵀJ + λI) Δc = Jᵀ residual
    """A = J.T @ J + np.eye(m) * 1e-8
    b = J.T @ residual
    delta = np.linalg.solve(A, b)"""

    # Solve normal equations: (JᵀJ + λI) Δc = Jᵀ residual
    A = J.T @ J + np.eye(m) * 1e-8
    b = J.T @ residual
    try:
        delta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Matrix is singular or ill-conditioned: skip update
        return best_vals.tolist(), worst_fitness, ind

    new_vals = best_vals + delta
    # Evaluate new fitness
    tmpn = copy.deepcopy(ind)
    tmpn = _apply_ephemerals(tmpn, new_vals.tolist(), eph)
    funcn = compile_tree(str(tmpn), pset)
    outn = funcn(*inputs)
    fitn = worst_fitness if outn.dim() != objective else fitness_fn(outn, y)

    return new_vals.tolist(), fitn, tmpn



def scale_data(X3d: Tensor, X2d: Tensor, y: Tensor):
    """
    Scale X3d, X2d, and y by subtracting mean and dividing by std per variable.
    Returns (X3d_scaled, X2d_scaled, y_scaled, stats) where stats is a dict containing
    'X3d_mean', 'X3d_std', 'X2d_mean', 'X2d_std', 'y_mean', 'y_std'.
    """
    # X2d
    X2d_mean = X2d.mean(dim=0)
    X2d_std = X2d.std(dim=0, unbiased=False)
    X2d_scaled = (X2d - X2d_mean) / X2d_std
    # X3d: shape (n_obs, n_vars3, time)
    # compute per-variable mean and std over samples and time
    X3d_mean = X3d.mean(dim=(0,2))
    X3d_std = X3d.std(dim=(0,2), unbiased=False)
    # reshape for broadcasting
    X3d_scaled = (X3d - X3d_mean[None,:,None]) / X3d_std[None,:,None]
    # y
    if y.dim() > 1:
        y_mean = y.mean(dim=(0, 1))
        y_std = y.std(dim=(0, 1), unbiased=False)
    else:
        y_mean = y.mean()
        y_std = y.std(unbiased=False)
    y_scaled = (y - y_mean) / y_std
    stats = {
        'X2d_mean': X2d_mean,
        'X2d_std': X2d_std,
        'X3d_mean': X3d_mean,
        'X3d_std': X3d_std,
        'y_mean': y_mean,
        'y_std': y_std,
    }
    return X3d_scaled, X2d_scaled, y_scaled, stats

def unscale_expression(expr: sp.Expr, stats: dict):
    """
    Unscale a sympy expression expr that was learned on scaled variables.
    Substitutes each symbol x{i} and v{i} by (symbol - mean)/std, then
    rescales the output via y = y_mean + y_std * expr.
    Returns a simplified sympy expression in the original units.
    """
    # build substitution dict for x symbols
    subs = {}
    for i, (mu, sigma) in enumerate(zip(stats['X2d_mean'], stats['X2d_std'])):
        sym = sp.symbols(f'x{i}')
        subs[sym] = (sym - float(mu)) / float(sigma)
    # build substitution dict for v symbols
    for i, (mu, sigma) in enumerate(zip(stats['X3d_mean'], stats['X3d_std'])):
        sym = sp.symbols(f'v{i}')
        subs[sym] = (sym - float(mu)) / float(sigma)
    # perform variable unscaling
    expr_unscaled = expr.subs(subs)
    # unscale output y
    y_mean = float(stats['y_mean'])
    y_std = float(stats['y_std'])
    expr_unscaled = y_mean + y_std * expr_unscaled
    return sp.simplify(expr_unscaled)