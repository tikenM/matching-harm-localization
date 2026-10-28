"""Case generators with integer data. All values are int or Fraction.

  integer rank-one path : integer features X (n x r), Y (m x r), score S = X Y^T, and the
                          rank-one attenuation x_i -> x_i - g (x_i.v) v / (v.v) for an
                          integer direction v, so D_ij = -(x_i.v)(y_j.v)/(v.v).
  fitted linear scorer  : integer features X, integer targets T on all pairs; the weights
                          w_j solve the normal equations (exact rational least squares),
                          S_ij = x_i . w_j, and the same attenuation of x_i gives
                          D_ij = -(x_i.v)(w_j.v)/(v.v).
Preferences: a common integer quality per opportunity plus an integer idiosyncratic term;
each applicant lists the opportunities they are eligible for in decreasing order.
"""
import random
from fractions import Fraction


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def solve(A, b):
    """Exact Gaussian elimination for A x = b with Fractions. Returns None if singular."""
    n = len(A)
    M = [[Fraction(x) for x in row] + [Fraction(bb)] for row, bb in zip(A, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]


def make_prefs(rng, n, m, deg, quality):
    pref = []
    for i in range(n):
        el = rng.sample(range(m), min(deg, m))
        key = {j: 4 * quality[j] + rng.randint(0, 9) for j in el}
        pref.append(sorted(el, key=lambda j: (-key[j], j)))
    return pref


def group_direction(X, group):
    r = len(X[0])
    g1 = [i for i in range(len(X)) if group[i] == 1]
    g0 = [i for i in range(len(X)) if group[i] == 0]
    v = [len(g0) * sum(X[i][k] for i in g1) - len(g1) * sum(X[i][k] for i in g0)
         for k in range(r)]
    return v


def make_case(seed, scorer, direction):
    rng = random.Random(seed)
    n = rng.randint(10, 24)
    m = rng.randint(3, 6)
    cap = [rng.randint(1, 3) for _ in range(m)]
    while sum(cap) >= n:
        cap[rng.randrange(m)] = max(1, cap[rng.randrange(m)] - 1)
        if sum(cap) == m:
            break
    r = 3
    deg = rng.randint(2, m)
    X = [[rng.randint(-5, 5) for _ in range(r)] for _ in range(n)]
    group = [rng.randint(0, 1) for _ in range(n)]
    for i in range(n):
        X[i][0] += 3 * group[i]
    Y = [[rng.randint(-5, 5) for _ in range(r)] for _ in range(m)]
    quality = [rng.randint(0, 5) for _ in range(m)]
    pref = make_prefs(rng, n, m, deg, quality)
    if scorer == "integer":
        W = [[Fraction(y) for y in row] for row in Y]
    else:
        W = []
        XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(r)] for a in range(r)]
        for j in range(m):
            T = [dot(X[i], Y[j]) + rng.randint(-8, 8) for i in range(n)]
            Xt = [sum(X[i][a] * T[i] for i in range(n)) for a in range(r)]
            w = solve(XtX, Xt)
            if w is None:
                return None
            W.append(w)
    if direction == "group":
        v = group_direction(X, group)
    else:
        v = [rng.randint(-4, 4) for _ in range(r)]
    vv = dot(v, v)
    if vv == 0:
        return None
    S = [[dot(X[i], W[j]) for j in range(m)] for i in range(n)]
    D = [[-Fraction(dot(X[i], v)) * dot(W[j], v) / vv for j in range(m)] for i in range(n)]
    return dict(pref=pref, cap=cap, S=S, D=D, n=n, m=m, scorer=scorer, direction=direction)
