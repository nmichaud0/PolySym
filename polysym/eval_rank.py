# ─── eval_rank.py ─────────────────────────────────────────────────────────────
from deap import gp

# rank tags
R1, R2 = 1, 2                   # 1‑D  / 2‑D tensors

# reducers that force 1‑D
_UNARY_TO_RANK = {
    "unary_mean"  : R1,
    "unary_std"   : R1,
    "unary_min"   : R1,
    "unary_max"   : R1,
    "unary_median": R1,
}

def _terminal_rank(node: gp.Primitive) -> int:
    """Rank for leaves:  x* → R1,  v* → R2,  constants → R1"""
    if node.name.startswith("x"):   # scalar per obs.
        return R1
    if node.name.startswith("v"):   # time‑series per obs.
        return R2
    return R1                       # randc or any constant

# --------------------------------------------------------------------------- #
def is_valid_tree(individual: gp.PrimitiveTree, objective_dim: int) -> bool:
    """
    Fast dimensionality check.

    Parameters
    ----------
    individual    : DEAP PrimitiveTree
    objective_dim : 1  (target y is [N])   or
                    2  (target y is [N,T])

    Returns
    -------
    bool  – True if tree’s output rank matches target, else False
    """
    if objective_dim not in (1, 2):
        raise ValueError("objective_dim must be 1 or 2")

    stack = []                        # rank stack

    # Walk the prefix list **from right to left** so children come first
    for node in reversed(individual):

        if node.arity == 0:           # Terminal --------------------------------
            stack.append(_terminal_rank(node))
            continue

        # Non‑terminal ----------------------------------------------------------
        if len(stack) < node.arity:   # not enough children → malformed
            return False

        # Pop children ranks
        child_ranks = [stack.pop() for _ in range(node.arity)]

        if node.name.startswith("binary_"):
            result_rank = max(child_ranks)           # broadcasting rule
        else:  # unary
            result_rank = _UNARY_TO_RANK.get(node.name, child_ranks[0])

        stack.append(result_rank)

    # At the end exactly one rank must remain
    if len(stack) != 1:
        return False

    target_rank = R2 if objective_dim == 2 else R1
    return stack[0] == target_rank
# ─────────────────────────────────────────────────────────────────────────────
