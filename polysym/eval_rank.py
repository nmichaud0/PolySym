# eval_rank.py
from deap import gp

R1, R2 = 1, 2            # rank tags (vector and matrix)

_UNARY_TO_RANK = {
    "unary_mean"  : R1,
    "unary_std"   : R1,
    "unary_min"   : R1,
    "unary_max"   : R1,
    "unary_median": R1,
}

def _terminal_rank(node: gp.Primitive) -> int:
    if node.name.startswith("x"): return R1
    if node.name.startswith("v"): return R2
    return 0                     # constants → rank-0

def is_valid_tree(ind, objective_dim: int) -> bool:
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
            if node.name in _UNARY_TO_RANK:      # reducer
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


def old_is_valid_tree(ind, objective_dim: int) -> bool:
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
            if node.name in _UNARY_TO_RANK:      # reducer
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