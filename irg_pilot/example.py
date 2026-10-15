"""Exact verification of the four-applicant construction in the text."""
from fractions import Fraction as F
from .core import (Market, deferred_acceptance, is_stable, mean_loss,
                   universal_bound, opportunity_proposing_da)
from .path import units
from .bounds import witness_bound

X, Y, Z, W = 0, 1, 2, 3


def build():
    S = [[5, 19, 1, 14], [13, 20, 3, 0], [12, 14, 0, 20], [2, 12, 7, 1]]
    D = [[9, 9, 6, 6], [3, 3, 2, 2], [12, 12, 8, 8], [3, 3, 2, 2]]
    pref = [[W, Z, Y, X], [Y, X, W, Z], [Y, W, X, Z], [Y, X, W, Z]]
    mk = Market(n=4, m=4, cap=[1] * 4, elig=[[True] * 4 for _ in range(4)],
                pref=pref, S=[[F(v) for v in r] for r in S],
                D=[[F(v) for v in r] for r in D])
    return mk


def verify():
    mk = build()
    a, b = F(1, 4), F(3, 4)
    m0 = deferred_acceptance(mk, F(0))
    nu = [Y, X, W, Z]
    us = units(mk, a, b)
    stable_all = all(is_stable(mk, nu, g) for _, g in us)
    L = [mean_loss(mk, m0, deferred_acceptance(mk, g)) for _, g in us]
    # Utilities in this example are 1, 3/4, 1/2, 1/4 by rank,
    # which differs from the generator's rank-linear map; recompute here.
    util = lambda i, j: F(4 - mk.pref[i].index(j), 4)
    bnd = [max(F(0), util(i, m0[i]) - util(i, nu[i])) for i in range(4)]
    U0 = sum(util(i, m0[i]) for i in range(4)) / 4
    Lmax = max(sum(max(F(0), util(i, m0[i]) - util(i, ma[i])) for i in range(4)) / 4
               for ma in (deferred_acceptance(mk, g) for _, g in us))
    opt = opportunity_proposing_da(mk, F(1, 2))
    return {
        "M0": m0,
        "crossings_in_I": [str(g) for k, g in us if k == "point"][1:-1],
        "witness_stable_on_I": stable_all,
        "witness_is_opportunity_optimal_at_1/2": opt == nu,
        "B(nu)": str(sum(bnd) / 4),
        "U0": str(U0),
        "max_realized_mean_loss": str(Lmax),
    }
