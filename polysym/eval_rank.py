from deap import gp

# rank tags
R1, R2 = 1, 2  # target ranks for 1-D and 2-D outputs

# reducers that force 1-D, but only valid if child is R2
_UNARY_TO_RANK = {
    "unary_mean"  : R1,
    "unary_std"   : R1,
    "unary_min"   : R1,
    "unary_max"   : R1,
    "unary_median": R1,
}

def _terminal_rank(node: gp.Primitive) -> int:
    """Rank for leaves: x* → R1, v* → R2, constants → R0"""
    if node.name.startswith("x"):
        return R1
    if node.name.startswith("v"):
        return R2
    return 0  # CONSTANTS are rank-0

def is_valid_tree(individual: gp.PrimitiveTree, objective_dim: int) -> bool:
    """
    Fast dimensionality check.

    individual    : DEAP PrimitiveTree
    objective_dim : 1  (target y is a vector [N]) or
                    2  (target y is a 2-D tensor [N,T])
    """
    if objective_dim not in (1, 2):
        raise ValueError("objective_dim must be 1 or 2")

    stack = []
    # Walk the prefix list from right to left so children come first
    for node in reversed(individual):

        # ── Terminal nodes ──────────────────────────────────────────────
        if node.arity == 0:
            stack.append(_terminal_rank(node))
            continue

        # ── Malformed check ─────────────────────────────────────────────
        if len(stack) < node.arity:
            return False

        # ── Pop child ranks ─────────────────────────────────────────────
        child_ranks = [stack.pop() for _ in range(node.arity)]

        # ── Decide output rank ──────────────────────────────────────────
        if node.arity > 1:
            # Multi-input primitives broadcast (e.g. add, mul)
            result_rank = max(child_ranks)

        elif node.arity == 1:
            # Unary operator
            child_r = child_ranks[0]
            if node.name in _UNARY_TO_RANK:
                # Reducers only on R2 inputs
                if child_r != R2:
                    return False
                result_rank = R1
            else:
                # Elementwise unary (neg, abs, etc.) preserves rank
                result_rank = child_r

        else:
            # Shouldn’t happen (arity==0 handled above)
            result_rank = child_ranks[0]

        stack.append(result_rank)

    # ── Final check ───────────────────────────────────────────────────
    if len(stack) != 1:
        return False

    target_rank = R2 if objective_dim == 2 else R1
    return stack[0] == target_rank