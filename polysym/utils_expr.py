from __future__ import annotations
from deap import gp
import re

# tokeniser -----------------------------------------------------------
_TOK = re.compile(r"\s*([\w\.\-]+|[,()])")
def _stream(expr: str):
    for m in _TOK.finditer(expr):
        yield m.group(1)
    yield None

# build helper tables once per pset ----------------------------------
def _var_types(pset):
    vt = {}
    for rtype, terms in pset.terminals.items():
        for t in terms:
            vt[t.name] = rtype
    return vt

def _prim_by_name(pset):
    tbl = {}
    for rtype, plist in pset.primitives.items():
        for prim in plist:           # each prim appears in exactly one rtype list
            tbl.setdefault(prim.name, []).append(prim)
    return tbl

def _name_to_terminal(pset):
    tbl = {}
    for terms in pset.terminals.values():
        for t in terms:
            tbl[t.name] = t
    return tbl

# helper: get scalar type of pset for numeric literals
def _scalar_type(pset):
    """Return the return‑type class for numeric literals (assume same as randc or first Scalar)."""
    for rtype, terms in pset.terminals.items():
        for t in terms:
            if t.name == "randc":
                return rtype
    # fall back to the first key (usually Scalar)
    return next(iter(pset.terminals))

# main recursive parser ----------------------------------------------
def _parse(tok, pset, vt, ptbl):
    word = next(tok)
    if word is None:
        raise ValueError("unexpected EOF")
    if word in ",()":
        raise ValueError(f"unexpected '{word}'")

    # terminals -------------------------------------------------------
    if word in vt:
        term = _name_to_terminal(pset)[word]
        return [term], vt[word]

    # primitives ------------------------------------------------------
    if word not in ptbl:
        # numeric literal → create a fresh scalar terminal on the fly
        try:
            const_val = float(word)
            scalar_tp = _scalar_type(pset)
            const_term = gp.Terminal(const_val, False, scalar_tp)
            return [const_term], scalar_tp
        except ValueError:
            raise ValueError(f"unknown symbol '{word}'")

    if next(tok) != "(":
        raise ValueError(f"expected '(' after '{word}'")

    kids, ctypes = [], []
    while True:
        child, rt = _parse(tok, pset, vt, ptbl)
        kids.append(child); ctypes.append(rt)
        sep = next(tok)
        if sep == ")": break
        if sep != ",":  raise ValueError("expected ','")

    # choose primitive with exact arg match
    for prim in ptbl[word]:
        if tuple(prim.args) == tuple(ctypes):
            flat = [prim]
            for ch in kids: flat.extend(ch)
            return flat, prim.ret

    raise ValueError(f"{word}{tuple(ctypes)} not in primitive set")

def parse_to_tree(expr: str, pset: gp.PrimitiveSetTyped) -> gp.PrimitiveTree:
    vt   = _var_types(pset)
    ptbl = _prim_by_name(pset)
    slab, r = _parse(_stream(expr), pset, vt, ptbl)
    if r != pset.ret:
        raise TypeError(f"expression returns {r.__name__}, expected {pset.ret.__name__}")
    return gp.PrimitiveTree(slab)