"""Tightness of closure-based harm bounds on the pilot generator.

For each generated market, edit direction, edit norm and interval [0, b],
computes the displacement closure R(b) of the localization proposition and
compares the bounds it gives with the exact sweep.
Usage: python closure_bounds.py --markets 30 --seed 0 --out closure_bounds.json
"""
import argparse, json, os, sys, statistics
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.environ.get("IRG_PILOT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from irg_pilot.core import deferred_acceptance, losses, utility, universal_bound, UNMATCHED
from irg_pilot.path import exact_sweep
from irg_pilot.generate import make_market, group_direction, rotate, attenuation_edit, set_edit

ZERO = Fraction(0)

def seed_incumbents(mk, m0, b):
    X = set()
    for k in range(mk.n):
        j = m0[k]
        if j == UNMATCHED:
            continue
        for i in range(mk.n):
            if i == k or not mk.elig[i][j] or m0[i] == j:
                continue
            if mk.prefers(i, j, m0[i]) and mk.above(k, i, j, ZERO) and mk.above(i, k, j, b):
                X.add(k)
    return X

def closure(mk, m0, seed, b):
    R = set(seed)
    changed = True
    while changed:
        changed = False
        for k in range(mk.n):
            j = m0[k]
            if k in R or j == UNMATCHED:
                continue
            for r in R:
                if (mk.elig[r][j] and m0[r] != j and not mk.prefers(r, j, m0[r])
                        and (mk.above(r, k, j, ZERO) or mk.above(r, k, j, b))):
                    R.add(k); changed = True; break
    return R

def proposer_enclosure(mk, b):
    """Possible-proposer enclosure P over [0, b] (equation of Objective 2)."""
    P = {(i, mk.pref[i][0]) for i in range(mk.n) if mk.pref[i]}
    changed = True
    while changed:
        changed = False
        for (i, j) in list(P):
            A = sum(1 for (k, jj) in P if jj == j and k != i
                    and (mk.above(k, i, j, ZERO) or mk.above(k, i, j, b)))
            if A >= mk.cap[j]:
                r = mk.rank_in_pref(i, j)
                if r + 1 < len(mk.pref[i]) and (i, mk.pref[i][r + 1]) not in P:
                    P.add((i, mk.pref[i][r + 1])); changed = True
    return P

def landing_bounds(mk, m0, P, b):
    """Per-applicant loss bound from the last enclosed proposal. Applicant i
    ends at an opportunity it proposed to, so its utility is at least that of
    the last enclosed one, unless the list is exhausted inside P and the last
    opportunity can reject (then 0 is possible)."""
    out = []
    for i in range(mk.n):
        u0 = utility(mk, i, m0[i])
        js = [j for j in mk.pref[i] if (i, j) in P]
        if not js:
            out.append(u0); continue
        last = js[-1]
        floor = utility(mk, i, last)
        if len(js) == len(mk.pref[i]):
            A = sum(1 for (k, jj) in P if jj == last and k != i
                    and (mk.above(k, i, last, ZERO) or mk.above(k, i, last, b)))
            if A >= mk.cap[last]:
                floor = Fraction(0)
        out.append(max(Fraction(0), u0 - floor))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--out", default="closure_bounds.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for mid in range(a.markets):
        mk, X, Y, group, elig = make_market(rng)
        S = X @ Y.T
        w = group_direction(X, group)
        dirs = {"fitted": w, "rotated_0.5": rotate(w, 0.5, rng), "random": rotate(w, np.pi / 2, rng)}
        for dn, wd in dirs.items():
            for rel in a.levels:
                set_edit(mk, attenuation_edit(X, Y, wd, elig, S, rel))
                m0 = deferred_acceptance(mk, ZERO)
                u0 = [utility(mk, i, m0[i]) for i in range(mk.n)]
                U0 = float(sum(u0) / mk.n)
                for b in (Fraction(1, 2), Fraction(1)):
                    us, allocs = exact_sweep(mk, ZERO, b)
                    per = [losses(mk, m0, al) for al in allocs]
                    losers = {i for x in per for i in range(mk.n) if x[i] > 0}
                    real_mean = float(max(sum(x) for x in per) / mk.n)
                    real_max = float(max(max(x) for x in per))
                    R = closure(mk, m0, seed_incumbents(mk, m0, b), b)
                    b_mean = float(sum(u0[i] for i in R) / mk.n)
                    b_max = float(max((u0[i] for i in R), default=0))
                    P = proposer_enclosure(mk, b)
                    lb = landing_bounds(mk, m0, P, b)
                    sound = all(lb[i] >= x[i] for x in per for i in range(mk.n))
                    sharp_max = float(max((lb[i] for i in R), default=0))
                    sharp_mean = float(sum(lb[i] for i in R) / mk.n)
                    elig_pairs = sum(len(p) for p in mk.pref)
                    rows.append(dict(market=mid, sharp_sound=sound, sharp_max=sharp_max,
                        sharp_mean=sharp_mean, P_frac=len(P) / elig_pairs, direction=dn, rel=rel, b=str(b),
                        closure_frac=len(R) / mk.n, n_losers=len(losers),
                        losers_in_closure=losers <= R, U0=U0,
                        bound_mean=b_mean, real_mean=real_mean,
                        bound_max=b_max, real_max=real_max))
    json.dump(rows, open(a.out, "w"))
    def med(xs): return statistics.median(xs) if xs else float("nan")
    print("intervals", len(rows), "all losers in closure:", all(r["losers_in_closure"] for r in rows))
    pos = [r for r in rows if r["n_losers"] > 0]
    print("positive-loss intervals", len(pos))
    for lab, sub in (("all", rows), ("positive-loss", pos)):
        print(f"[{lab}] n={len(sub)}")
        print("  median closure share of applicants", round(med([r['closure_frac'] for r in sub]), 3))
        print("  share with empty closure", round(sum(r['closure_frac'] == 0 for r in sub) / len(sub), 3))
        print("  median mean-loss bound / U0", round(med([r['bound_mean'] / r['U0'] for r in sub]), 3))
        print("  share with mean-loss bound <= 0.5 U0", round(sum(r['bound_mean'] <= 0.5 * r['U0'] for r in sub) / len(sub), 3))
    sp = [r for r in pos if r['real_mean'] > 0]
    print("positive-loss: median bound_mean / realized worst mean loss", round(med([r['bound_mean'] / r['real_mean'] for r in sp]), 2))
    print("positive-loss: median bound_max", round(med([r['bound_max'] for r in pos]), 3),
          "median realized max loss", round(med([r['real_max'] for r in pos]), 3))
    print("share of all intervals with bound_max <= 0.6:", round(sum(r['bound_max'] <= 0.6 for r in rows) / len(rows), 3))
    print("sharper landing bound sound in all intervals:", all(r["sharp_sound"] for r in rows))
    print("proposer enclosure: median share of eligible pairs", round(med([r["P_frac"] for r in rows]), 3))
    print("positive-loss: median sharper max-loss bound", round(med([r["sharp_max"] for r in pos]), 3),
          "(closure-only", round(med([r["bound_max"] for r in pos]), 3), ", realized", round(med([r["real_max"] for r in pos]), 3), ")")
    print("positive-loss: median sharper mean-loss bound / realized", round(med([r["sharp_mean"] / r["real_mean"] for r in sp]), 2),
          "(closure-only", round(med([r["bound_mean"] / r["real_mean"] for r in sp]), 2), ")")
    print("positive-loss: share with sharper max-loss bound <= 0.6:", round(sum(r["sharp_max"] <= 0.6 for r in pos) / len(pos), 3),
          " closure-only:", round(sum(r["bound_max"] <= 0.6 for r in pos) / len(pos), 3))
    for rel in a.levels:
        sub = [r for r in rows if r['rel'] == rel]
        print(f"rel={rel}: median closure share {round(med([r['closure_frac'] for r in sub]),3)}, share empty {round(sum(r['closure_frac']==0 for r in sub)/len(sub),3)}")

if __name__ == "__main__":
    main()
