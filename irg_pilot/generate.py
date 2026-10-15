"""Synthetic markets with learned-style bilinear scores and a protected
group, and rank-one attenuation edits along chosen directions."""
from fractions import Fraction
import numpy as np
from .core import Market

DENOM = 10 ** 6  # scores are rounded to this rational grid, then exact


def _frac(x: float) -> Fraction:
    return Fraction(int(round(x * DENOM)), DENOM)


def make_market(rng, n=60, m=15, cap=3, deg=4, r=6, shift=0.8):
    """Applicant embeddings X (n x r) carry a group shift along a hidden
    direction; opportunity embeddings Y (m x r). Scores S = X Y^T.
    Each applicant is eligible at `deg` random opportunities and ranks them
    in random order. Returns the market (with D unset), X, Y and groups."""
    group = rng.integers(0, 2, size=n)
    w_true = rng.normal(size=r)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(n, r)) + shift * np.outer(group - 0.5, w_true)
    Y = rng.normal(size=(m, r))
    S = X @ Y.T
    elig = np.zeros((n, m), bool)
    for i in range(n):
        elig[i, rng.choice(m, deg, replace=False)] = True
    pref = [list(rng.permutation(np.where(elig[i])[0]).tolist()) for i in range(n)]
    mk = Market(n=n, m=m, cap=[cap] * m, elig=elig.tolist(), pref=pref,
                S=[[_frac(v) for v in row] for row in S], D=None,
                group=group.tolist())
    return mk, X, Y, group, elig


def group_direction(X, group):
    w = X[group == 1].mean(0) - X[group == 0].mean(0)
    return w / np.linalg.norm(w)


def rotate(w, angle, rng):
    """Rotate unit vector w by `angle` radians towards a random orthogonal
    direction."""
    z = rng.normal(size=w.shape)
    z -= (z @ w) * w
    z /= np.linalg.norm(z)
    return np.cos(angle) * w + np.sin(angle) * z


def attenuation_edit(X, Y, w, elig, S_float, rel):
    """Rank-one attenuation along w: X -> X - g (X w) w^T, so the score
    change at full strength is D = -(X w)(Y w)^T, rescaled so that its
    Frobenius norm over eligible entries is `rel` times that of S."""
    D = -np.outer(X @ w, Y @ w)
    scale = rel * np.linalg.norm(S_float[elig]) / np.linalg.norm(D[elig])
    return D * scale


def set_edit(mk, D):
    mk.D = [[_frac(v) for v in row] for row in D]
    return mk
