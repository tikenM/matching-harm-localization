"""Independent verifier component: round-based DA written without reference to
trace_sweep.py. All unmatched applicants propose simultaneously in each round; each
opportunity keeps its best `cap` proposals among those held and new. Uses only
fractions.Fraction and the same input conventions (pref, cap, S, D, ties to lower index).
"""
from fractions import Fraction


def round_da(pref, cap, S, D, g):
    n, m = len(pref), len(cap)
    pos = [0] * n                    # next list position for each applicant
    holder = [-1] * n                # opportunity currently holding the applicant
    while True:
        proposals = {}
        for i in range(n):
            if holder[i] == -1 and pos[i] < len(pref[i]):
                proposals.setdefault(pref[i][pos[i]], []).append(i)
                pos[i] += 1
        if not proposals:
            break
        for j, new in proposals.items():
            pool = [i for i in range(n) if holder[i] == j] + new
            pool.sort(key=lambda a: (-(S[a][j] + g * D[a][j]), a))
            for rank, a in enumerate(pool):
                holder[a] = j if rank < cap[j] else -1
    return holder
