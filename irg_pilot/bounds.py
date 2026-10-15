"""Witness bounds and greedy piecewise covers over an interval."""
from fractions import Fraction
from typing import List
from .core import (Market, deferred_acceptance, opportunity_proposing_da,
                   is_stable, losses)


def witness_bound(mk: Market, m0, nu) -> List[Fraction]:
    """Individual bounds b_i(nu) = max(0, u_i(M0(i)) - u_i(nu(i)))."""
    return losses(mk, m0, nu)


def greedy_cover(mk: Market, m0, us, checks=None):
    """Cover the units of an interval by pieces. Each piece is a contiguous
    run of units with one witness stable at the representatives of its first
    and last unit; by affinity of every priority comparison it is then
    stable on the whole run. Candidate witnesses at the start of a piece are
    the applicant- and opportunity-proposing DA outputs there; the candidate
    that extends furthest is kept.

    `checks`, if given, is a one-element list incremented once per
    stability check. Returns a list of (start_unit, end_unit, witness)."""
    if checks is None:
        checks = [0]
    pieces = []
    s = 0
    K = len(us)
    while s < K:
        g0 = us[s][1]
        best = None
        for nu in (deferred_acceptance(mk, g0), opportunity_proposing_da(mk, g0)):
            checks[0] += 1
            if not is_stable(mk, nu, g0):
                continue
            # The set of strengths at which nu is stable is an interval,
            # because every stability condition is affine in the strength.
            # The furthest unit is therefore found by binary search.
            lo, hi = s, K - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                checks[0] += 1
                if is_stable(mk, nu, us[mid][1]):
                    lo = mid
                else:
                    hi = mid - 1
            t = lo
            if best is None or t > best[1]:
                best = (s, t, nu)
        pieces.append(best)
        s = best[1] + 1
    return pieces


def piecewise_bound(mk: Market, m0, pieces) -> List[Fraction]:
    per = [witness_bound(mk, m0, nu) for _, _, nu in pieces]
    return [max(p[i] for p in per) for i in range(mk.n)]
