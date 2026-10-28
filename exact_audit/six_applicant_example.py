"""Search for and verify a small example: six applicants, two opportunities of capacity two,
three distinct matched sets along the path, and no assignment that is stable at every
strength in [0,1]. The record is verified by verify_record.verify.

  python six_applicant_example.py [--max-seed 20000]

The search takes the first seed (in increasing order) whose sweep has exactly three
distinct matched sets and exactly four outcome pieces (maximal intervals with one
assignment). The data are integers.
"""
import argparse, itertools, json, random
from fractions import Fraction
from trace_sweep import Market, sweep, all_crossings
from verify_record import verify

ZERO, ONE = Fraction(0), Fraction(1)


def stable(pref, cap, S, D, match, g):
    n, m = len(pref), len(cap)
    for i in range(n):
        if match[i] != -1 and match[i] not in pref[i]:
            return False
    for j in range(m):
        if sum(1 for i in range(n) if match[i] == j) > cap[j]:
            return False
    key = lambda i, j: (S[i][j] + g * D[i][j], -i)
    for i in range(n):
        cur = match[i]
        for j in pref[i]:
            if cur != -1 and pref[i].index(j) >= pref[i].index(cur):
                break
            held = [k for k in range(n) if match[k] == j]
            if len(held) < cap[j] or key(i, j) > min(key(k, j) for k in held):
                return False
    return True


def build(seed):
    rng = random.Random(seed)
    n, m = 6, 2
    pref = []
    for i in range(n):
        pref.append([0, 1] if rng.random() < 0.5 else [1, 0])
    S = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
    D = [[Fraction(rng.randint(-12, 12)) for _ in range(m)] for _ in range(n)]
    return pref, [2, 2], S, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seed", type=int, default=20000)
    ap.add_argument("--out", default="six_applicant_example.json")
    a = ap.parse_args()
    for seed in range(a.max_seed):
        pref, cap, S, D = build(seed)
        mk = Market(pref, cap, S, D)
        rec = sweep(mk)
        sets = {frozenset(i for i, j in enumerate(p["match"]) if j != -1) for p in rec}
        merged = []
        for p in rec:
            if not merged or merged[-1] != p["match"]:
                merged.append(p["match"])
        if len(sets) == 3 and len(merged) == 4:
            ok, why = verify(pref, cap, S, D, rec)
            cr = all_crossings(mk)
            pts = [ZERO, ONE] + cr
            units = pts + [(x + y) / 2 for x, y in zip(sorted(pts), sorted(pts)[1:])]
            common = [M for M in itertools.product([-1, 0, 1], repeat=6)
                      if all(stable(pref, cap, S, D, list(M), g) for g in units)]
            out = dict(seed=seed, pref=pref, cap=cap, S=S, D=[[str(x) for x in r] for r in D],
                       trace_pieces=len(rec), outcome_pieces=len(merged),
                       matched_sets=[sorted(s) for s in sets], verified=ok, reason=why,
                       assignments_stable_at_every_unit=len(common),
                       pieces=[dict(lo=str(p["lo"]), lo_closed=p["lo_closed"], hi=str(p["hi"]),
                                    hi_closed=p["hi_closed"], match=p["match"]) for p in rec])
            print(json.dumps(out, indent=1))
            json.dump(out, open(a.out, "w"), indent=1)
            return
    print("no example found")


if __name__ == "__main__":
    main()
