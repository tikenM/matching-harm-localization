"""Seed sets, displacement closure and starting-class conditions."""
from fractions import Fraction
from typing import List, Set
from .core import Market, UNMATCHED, utility


def seed_all_pairs(mk: Market, b: Fraction) -> Set[int]:
    """Applicants in any eligible pair whose order differs between 0 and b."""
    X = set()
    for j in range(mk.m):
        el = [i for i in range(mk.n) if mk.elig[i][j]]
        for p in range(len(el)):
            for q in range(p + 1, len(el)):
                i, k = el[p], el[q]
                if mk.above(i, k, j, Fraction(0)) != mk.above(i, k, j, b):
                    X |= {i, k}
    return X


def seed_blocking(mk: Market, m0: List[int], b: Fraction) -> Set[int]:
    """Applicants in blocking-relevant pairs: incumbent k at j under M0,
    challenger i eligible at j who prefers j to M0(i), with k above i at 0
    and i above k at b. These are exactly the pairs that can block M0 at
    some strength in [0, b]."""
    X = set()
    for k in range(mk.n):
        j = m0[k]
        if j == UNMATCHED:
            continue
        for i in range(mk.n):
            if i == k or not mk.elig[i][j] or m0[i] == j:
                continue
            if not mk.prefers(i, j, m0[i]):
                continue
            if mk.above(k, i, j, Fraction(0)) and mk.above(i, k, j, b):
                X |= {i, k}
    return X


def closure(mk: Market, m0: List[int], seed: Set[int], b: Fraction) -> Set[int]:
    """Add any M0 incumbent whose priority, at 0 or b, is below that of a
    current member at an opportunity where the member is eligible."""
    R = set(seed)
    changed = True
    while changed:
        changed = False
        for k in range(mk.n):
            j = m0[k]
            if k in R or j == UNMATCHED:
                continue
            for r in R:
                if mk.elig[r][j] and (mk.above(r, k, j, Fraction(0))
                                      or mk.above(r, k, j, b)):
                    R.add(k)
                    changed = True
                    break
    return R


def min_margin(mk: Market, m0: List[int], R: Set[int], b: Fraction):
    """Smallest score margin by which an incumbent outside R exceeds an
    eligible member of R, over gamma in {0, b}. None if no such pair."""
    best = None
    for k in range(mk.n):
        j = m0[k]
        if k in R or j == UNMATCHED:
            continue
        for r in R:
            if not mk.elig[r][j]:
                continue
            for g in (Fraction(0), b):
                d = mk.score(k, j, g) - mk.score(r, j, g)
                best = d if best is None else min(best, d)
    return best


def utility_share(mk: Market, m0: List[int], R: Set[int]) -> Fraction:
    """Share of original utility held inside R (left side of C3 divided by U0)."""
    tot = sum(utility(mk, i, m0[i]) for i in range(mk.n))
    if tot == 0:
        return Fraction(0)
    return sum(utility(mk, i, m0[i]) for i in R) / tot
