# eval_rank.py
from deap import gp
from polysym.torch_operators_2 import (UNARY_NONREDUCE, UNARY_REDUCE, BINARY_NONREDUCE, BINARY_REDUCE)
import re

R0, R1, R2 = 0, 1, 2            # rank tags (vector and matrix)

_UNARY_TO_RANK = {
    "unary_mean"  : R1,
    "unary_std"   : R1,
    "unary_min"   : R1,
    "unary_max"   : R1,
    "unary_median": R1,
}

reducers = list(UNARY_REDUCE.keys()) + list(BINARY_REDUCE.keys())

class DimMismatch(Exception):
    """Exception"""

def _terminal_rank(node: gp.Primitive) -> int:
    if node.name.startswith("x"): return R1
    if node.name.startswith("v"): return R2
    return R0  # constant


class TreeNode:
    def __init__(self, prim):
        self.prim = prim
        self.children = []


class TreeNode2:
    def __init__(self, name, children=None):
        self.name     = name           # e.g. "binary_add" or "v0" or "3.14"
        self.children = children or [] # list of TreeNode

    def __repr__(self):
        if not self.children:
            return self.name
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.name}({args})"

# ────────── tokenizer ──────────
_TOKEN_RE = re.compile(r"""
    (?P<NAME>[A-Za-z_]\w*)     |   # identifiers: unary_add, x0, v1, etc
    (?P<NUMBER>-?\d+(\.\d+)?)  |   # numeric literals
    (?P<LPAR>\()               |   # left paren
    (?P<RPAR>\))               |   # right paren
    (?P<COMMA>,)                   # comma
""", re.VERBOSE)

def tokenize(s):
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        val  = m.group()
        yield kind, val
    yield None, None  # sentinel



def build_tree(ind):
    it = iter(ind)

    def recurse():
        node = next(it)  # take one gp.Primitive/Terminal
        tree = TreeNode(node)
        for _ in range(node.arity):  # grab exactly `arity` children
            tree.children.append(recurse())
        return tree

    return recurse()

def infer_vectorality(node: TreeNode) -> bool:
    """
    Returns True if the subtree outputs a vector (v* without intervening reducer),
    False otherwise.
    """
    # Terminal?
    if node.prim.arity == 0:
        name = getattr(node.prim, "name", "")
        return name.startswith("v")

    # Recurse on children
    child_vecs = [infer_vectorality(ch) for ch in node.children]
    opname     = node.prim.name

    # Unary
    if node.prim.arity == 1:
        op = opname[len("unary_"):]
        if op in UNARY_REDUCE:
            return False
        return child_vecs[0]

    # Binary
    op = opname[len("binary_"):]
    if op in BINARY_REDUCE:
        return False
    return any(child_vecs)

def is_valid_tree(ind, objective):

    root = build_tree(ind)
    out_is_vec = infer_vectorality(root)
    need_vec = objective == 2
    return out_is_vec == need_vec

_UNARY_IS_REDUCER  = {f"unary_{k}"  for k in UNARY_REDUCE}
_BINARY_IS_REDUCER = {f"binary_{k}" for k in BINARY_REDUCE}

def old_is_valid_tree(individual: gp.PrimitiveTree, objective_dim: int) -> bool:
    """
    Same behaviour you had before **plus**:
        • understands every reducer in UNARY_REDUCE / BINARY_REDUCE
        • never forbids broadcasts that PyTorch accepts
    """
    if objective_dim not in (1, 2):
        raise ValueError("objective_dim must be 1 or 2")

    stack        = []
    seen_R2_leaf = False                     # for the R2-must-return-R2 rule

    for node in reversed(individual):       # children first (post-order)
        if node.arity == 0:                               # TERMINAL
            r = _terminal_rank(node)
            seen_R2_leaf |= (r == R2)
            stack.append(r)
            continue

        if len(stack) < node.arity:         # malformed GP expression
            return False

        # ---------- UNARY ------------------------------------------------
        if node.arity == 1:
            child_r = stack.pop()
            if node.name in _UNARY_IS_REDUCER:
                if child_r != R2:           # reducers must eat vectors
                    return False
                stack.append(R1)            # reduce → per-obs scalar
            else:
                stack.append(child_r)       # element-wise unary
            continue

        # ---------- BINARY (arity == 2) ----------------------------------
        right_r = stack.pop()
        left_r  = stack.pop()

        if node.name in _BINARY_IS_REDUCER:
            # reducers defined so far (cov, pearson, …) require vectors
            if left_r != R2 or right_r != R2:
                return False
            stack.append(R1)
        else:
            # element-wise binary  → broadcast semantics
            if R0 in (left_r, right_r):                 # constant with anything
                stack.append(right_r if left_r == R0 else left_r)
            else:
                stack.append(max(left_r, right_r))      # R1 vs R2 → R2

    # ---------- final consistency check -------------------------------
    if len(stack) != 1:
        return False

    final_r = stack[0]
    if objective_dim == 2 and seen_R2_leaf:
        return final_r == R2          # must return a vector
    return final_r == (R2 if objective_dim == 2 else R1)




def oldis_valid_tree(ind, objective_dim: int) -> bool:
    if objective_dim not in (1, 2):
        raise ValueError("objective_dim must be 1 or 2")

    stack          = []
    seen_R2_leaf   = False       # ← new flag

    for node in reversed(ind):   # prefix walk, children first
        if node.arity == 0:                      # ── terminals ──
            r = _terminal_rank(node)
            seen_R2_leaf |= (r == R2)            # remember
            stack.append(r)
            continue

        if len(stack) < node.arity:              # malformed
            return False

        child_ranks = [stack.pop() for _ in range(node.arity)]

        if node.arity > 1:                       # binary / n-ary
            result_rank = max(child_ranks)
        else:                                    # unary
            child_r = child_ranks[0]
            if node.name in reducers:      # reducer
                if child_r != R2:                # only on matrices
                    return False
                result_rank = R1
            else:                                # elem-wise unary
                result_rank = child_r

        stack.append(result_rank)

    if len(stack) != 1:          # more than one item left ⇒ bad
        return False

    final_rank = stack[0]
    if objective_dim == 2 and seen_R2_leaf:
        # tree touches a v*  →  must end as R2
        return final_rank == R2

    # otherwise fall back to simple match (old behaviour)
    target_rank = R2 if objective_dim == 2 else R1
    return final_rank == target_rank