"""Independent checker of a sweep record.

Checks
  1. Tiling: pieces cover [0,1] with no gap or overlap and consistent closure flags.
  2. Exhaustive agreement: at every crossing strength and at one strength strictly
     between each pair of consecutive crossings, the assignment of the piece covering
     that strength equals the output of round_da (all-pairs crossings, recomputed here).
  3. Certificates: every recorded comparison (j, winner, loser) holds on the whole
     piece (the winner-beats-loser set contains the piece interval).
Returns (ok, reason).
"""
from fractions import Fraction
from round_da import round_da

ZERO, ONE = Fraction(0), Fraction(1)


def _covers(p, g):
    lo_ok = g > p["lo"] or (g == p["lo"] and p["lo_closed"])
    hi_ok = g < p["hi"] or (g == p["hi"] and p["hi_closed"])
    return lo_ok and hi_ok


def _crossings(pref, S, D, m):
    pts = set()
    n = len(pref)
    for j in range(m):
        el = [i for i in range(n) if j in pref[i]]
        for a in range(len(el)):
            for b in range(a + 1, len(el)):
                x, y = el[a], el[b]
                dd = D[x][j] - D[y][j]
                if dd != 0:
                    r = -(S[x][j] - S[y][j]) / dd
                    if ZERO < r < ONE:
                        pts.add(r)
    return sorted(pts)


def _holds_on_piece(S, D, w, l, j, p):
    """The comparison 'w beats l at j' holds at every strength in the piece."""
    A = S[w][j] - S[l][j]
    B = D[w][j] - D[l][j]
    ends = []
    if p["lo"] is not None:
        ends.append((p["lo"], p["lo_closed"]))
    if p["hi"] is not None:
        ends.append((p["hi"], p["hi_closed"]))
    # f(g) = A + B g is linear, so it suffices to test the two endpoints, and for an
    # open endpoint, the value at the endpoint must not be strictly in the wrong direction.
    for g, closed in ends:
        f = A + B * g
        if f > 0:
            continue
        if f == 0 and (w < l or not closed):
            # a tie at an open endpoint is not part of the piece; at a closed endpoint
            # the tie rule must favour the winner
            continue
        return False
    # interior: linear function, endpoints agree, so interior agrees (f cannot change
    # sign strictly inside without being wrong at an end or tying at an open end only)
    mid = (p["lo"] + p["hi"]) / 2
    f = A + B * mid
    return f > 0 or (f == 0 and w < l)


def verify(pref, cap, S, D, pieces):
    m = len(cap)
    if not pieces:
        return False, "empty record"
    if pieces[0]["lo"] != ZERO or not pieces[0]["lo_closed"]:
        return False, "does not start at 0 (closed)"
    if pieces[-1]["hi"] != ONE or not pieces[-1]["hi_closed"]:
        return False, "does not end at 1 (closed)"
    for a, b in zip(pieces, pieces[1:]):
        if a["hi"] != b["lo"]:
            return False, "gap or overlap between pieces"
        if a["hi_closed"] == b["lo_closed"]:
            return False, "closure flags do not complement"
    for p in pieces:
        if p["lo"] > p["hi"] or (p["lo"] == p["hi"] and not (p["lo_closed"] and p["hi_closed"])):
            return False, "malformed interval"
    cr = [ZERO] + _crossings(pref, S, D, m) + [ONE]
    samples = list(cr)
    samples += [(cr[k] + cr[k + 1]) / 2 for k in range(len(cr) - 1)]
    for g in samples:
        covering = [p for p in pieces if _covers(p, g)]
        if len(covering) != 1:
            return False, "strength %s covered %d times" % (g, len(covering))
        if round_da(pref, cap, S, D, g) != covering[0]["match"]:
            return False, "assignment differs from round-based DA at %s" % g
    for p in pieces:
        for (j, w, l) in p["trace"]:
            if not _holds_on_piece(S, D, w, l, j, p):
                return False, "recorded comparison does not hold on piece"
    return True, "ok"
