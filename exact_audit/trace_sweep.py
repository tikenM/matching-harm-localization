"""Exact comparison-trace sweep of applicant-proposing deferred acceptance (DA)
along a linear priority path S_g = S + g * D, 0 <= g <= 1.

All arithmetic uses fractions.Fraction. The code records, for every rejection made by
a DA run, the comparison that justified it. The recorded comparisons are valid on an
interval of strengths, and DA returns the same assignment on that interval. The sweep
restarts at the end of each interval. Output: a list of pieces (interval, assignment,
recorded comparisons).

Conventions
  applicant i, opportunity j; pref[i] is i's strict list of acceptable opportunities;
  cap[j] is the capacity; priority of i at j is S[i][j] + g * D[i][j]; ties are broken
  in favour of the lower applicant index. An assignment is a list: match[i] = j or -1.
"""
from fractions import Fraction
from collections import deque

ZERO, ONE = Fraction(0), Fraction(1)


class Market:
    def __init__(self, pref, cap, S, D):
        self.n, self.m = len(pref), len(cap)
        self.pref, self.cap, self.S, self.D = pref, cap, S, D

    def key(self, i, j, g):
        return self.S[i][j] + g * self.D[i][j]

    def beats(self, a, b, j, g):
        """True if a has priority over b at opportunity j at strength g."""
        ka, kb = self.key(a, j, g), self.key(b, j, g)
        return ka > kb or (ka == kb and a < b)


# An interval is (lo, lo_closed, hi, hi_closed) with Fraction ends (None = unbounded).
def beats_set(mk, w, l, j):
    """Set of strengths g in R at which w beats l at j, as an interval."""
    A = mk.S[w][j] - mk.S[l][j]
    B = mk.D[w][j] - mk.D[l][j]
    if B == 0:
        if A > 0 or (A == 0 and w < l):
            return (None, False, None, False)
        return "empty"
    r = -A / B
    tie_ok = w < l
    if B > 0:
        return (r, tie_ok, None, False)
    return (None, False, r, tie_ok)


def intersect(I, J):
    """Intersection of two intervals (or 'empty')."""
    if I == "empty" or J == "empty":
        return "empty"
    lo, loc, hi, hic = I
    lo2, loc2, hi2, hic2 = J
    if lo2 is not None and (lo is None or lo2 > lo or (lo2 == lo and not loc2)):
        lo, loc = lo2, loc2
    elif lo2 is not None and lo2 == lo:
        loc = loc and loc2
    if hi2 is not None and (hi is None or hi2 < hi or (hi2 == hi and not hic2)):
        hi, hic = hi2, hic2
    elif hi2 is not None and hi2 == hi:
        hic = hic and hic2
    if lo is not None and hi is not None:
        if lo > hi or (lo == hi and not (loc and hic)):
            return "empty"
    return (lo, loc, hi, hic)


def traced_da(mk, g):
    """Applicant-proposing DA at strength g. Returns (match, trace) where trace is a list
    of (j, winner, loser): each rejection of `loser` at j was justified by `winner`
    having priority over it. For a rejection, the loser is the lowest-priority applicant
    among those held plus the new proposer, so one comparison against every other member
    of that set is recorded."""
    nxt = [0] * mk.n
    held = [[] for _ in range(mk.m)]
    queue = deque(range(mk.n))
    trace = []
    while queue:
        i = queue.popleft()
        if nxt[i] >= len(mk.pref[i]):
            continue
        j = mk.pref[i][nxt[i]]
        nxt[i] += 1
        group = held[j] + [i]
        if len(group) <= mk.cap[j]:
            held[j] = group
            continue
        loser = group[0]
        for x in group[1:]:
            if mk.beats(loser, x, j, g):
                loser = x
        for x in group:
            if x != loser:
                trace.append((j, x, loser))
        held[j] = [x for x in group if x != loser]
        queue.append(loser)
    match = [-1] * mk.n
    for j in range(mk.m):
        for i in held[j]:
            match[i] = j
    return match, trace


def piece_at(mk, g):
    """Maximal interval within [0,1] on which the trace recorded at g stays valid."""
    match, trace = traced_da(mk, g)
    I = (ZERO, True, ONE, True)
    for (j, w, l) in trace:
        I = intersect(I, beats_set(mk, w, l, j))
    assert I != "empty"
    return I, match, trace


def all_crossings(mk):
    """Distinct strengths in (0,1) at which two applicants tie at some opportunity.
    Used only to choose a representative strength after a closed interval end."""
    pts = set()
    for j in range(mk.m):
        el = [i for i in range(mk.n) if j in mk.pref[i]]
        for p in range(len(el)):
            for q in range(p + 1, len(el)):
                a, b = el[p], el[q]
                B = mk.D[a][j] - mk.D[b][j]
                if B != 0:
                    r = -(mk.S[a][j] - mk.S[b][j]) / B
                    if ZERO < r < ONE:
                        pts.add(r)
    return sorted(pts)


def sweep(mk):
    """Exact sweep of [0,1]. Returns the list of pieces."""
    cross = all_crossings(mk)
    pieces = []
    t = ZERO
    while True:
        I, match, trace = piece_at(mk, t)
        pieces.append(dict(lo=I[0], lo_closed=I[1], hi=I[2], hi_closed=I[3],
                           match=match, trace=trace))
        lo, loc, hi, hic = I
        if hi == ONE and hic:
            return pieces
        if not hic:
            t = hi
        else:
            nxt = [c for c in cross if c > hi]
            t = (hi + (nxt[0] if nxt else ONE)) / 2
