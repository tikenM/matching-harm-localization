"""Priority crossings, priority regions and exact sweeps along S + g D."""
from fractions import Fraction
from typing import List, Tuple
from .core import Market, deferred_acceptance


def crossings(mk: Market, a: Fraction, b: Fraction) -> List[Fraction]:
    """Strengths strictly inside (a, b) at which two eligible applicants'
    scores at some opportunity become equal."""
    pts = set()
    for j in range(mk.m):
        el = [i for i in range(mk.n) if mk.elig[i][j]]
        for p in range(len(el)):
            for q in range(p + 1, len(el)):
                i, k = el[p], el[q]
                d0 = mk.S[i][j] - mk.S[k][j]
                dd = mk.D[i][j] - mk.D[k][j]
                if dd == 0:
                    continue
                g = -d0 / dd
                if a < g < b:
                    pts.add(g)
    return sorted(pts)


def units(mk: Market, a: Fraction, b: Fraction) -> List[Tuple[str, Fraction]]:
    """Points and open regions covering [a, b], each with a representative
    strength. Priorities are constant within each open region."""
    pts = [a] + crossings(mk, a, b) + [b]
    out = []
    for t in range(len(pts)):
        out.append(("point", pts[t]))
        if t + 1 < len(pts):
            out.append(("region", (pts[t] + pts[t + 1]) / 2))
    return out


def exact_sweep(mk: Market, a: Fraction, b: Fraction):
    """DA output at every unit representative of [a, b]."""
    us = units(mk, a, b)
    allocs = [deferred_acceptance(mk, g) for _, g in us]
    return us, allocs
