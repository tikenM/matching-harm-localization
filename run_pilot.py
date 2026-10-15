"""Pilot experiments.

Checks, on synthetic markets:
  (1) size of the displacement closure under the all-pairs seed and under
      the blocking-relevant seed;
  (2) whether every applicant with positive loss lies inside the closure
      (empirical localization; not a proof);
  (3) H2-style coverage: qualifying intervals and informative bounds from a
      single witness or a piecewise cover with fewer pieces than regions;
  (4) H1-style comparison of global and local features for predicting mean
      loss on held-out markets;
  (5) exact verification of the four-applicant construction.

Usage: python run_pilot.py --markets 60 --seed 0 --out results.json
"""
import argparse
import json
import time
from fractions import Fraction
import numpy as np

from irg_pilot.core import (deferred_acceptance, losses, universal_bound,
                            group_gap, UNMATCHED)
from irg_pilot.path import exact_sweep
from irg_pilot.closure import (seed_all_pairs, seed_blocking, closure,
                               min_margin, utility_share)
from irg_pilot.bounds import greedy_cover, piecewise_bound
from irg_pilot.generate import (make_market, group_direction, rotate,
                                attenuation_edit, set_edit)
from irg_pilot import example

INTERVALS = [(Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1)),
             (Fraction(0), Fraction(1))]


def analyse_interval(mk, m0, a, b, informative_ratio, kappa_frac, lam):
    us, allocs = exact_sweep(mk, a, b)
    n_regions = sum(1 for k, _ in us if k == "region")
    per_unit_losses = [losses(mk, m0, al) for al in allocs]
    L = [sum(x) / mk.n for x in per_unit_losses]
    losers = {i for x in per_unit_losses for i in range(mk.n) if x[i] > 0}
    distinct = len({tuple(al) for al in allocs})
    U0 = universal_bound(mk, m0)

    X_all = seed_all_pairs(mk, b)
    R_all = closure(mk, m0, X_all, b)
    X_blk = seed_blocking(mk, m0, b)
    R_blk = closure(mk, m0, X_blk, b)

    checks = [0]
    pieces = greedy_cover(mk, m0, us, checks)
    bnd = piecewise_bound(mk, m0, pieces)
    B = sum(bnd) / mk.n
    valid = all(bnd[i] >= max(x[i] for x in per_unit_losses) for i in range(mk.n))

    mu = min_margin(mk, m0, R_blk, b)
    share = utility_share(mk, m0, R_blk)
    in_class = (len(R_blk) <= kappa_frac * mk.n and share <= lam
                and (mu is None or mu > 0))
    qualifying = in_class and distinct >= 2 and len(losers) > 0
    compressed = len(pieces) == 1 or len(pieces) < n_regions
    informative = (U0 > 0 and B <= informative_ratio * U0 and compressed)
    return {
        "interval": [str(a), str(b)],
        "regions": n_regions,
        "distinct_allocations": distinct,
        "max_mean_loss": float(max(L)),
        "U0": float(U0),
        "n_losers": len(losers),
        "seed_all_frac": len(X_all) / mk.n,
        "closure_all_frac": len(R_all) / mk.n,
        "seed_blocking_frac": len(X_blk) / mk.n,
        "closure_blocking_frac": len(R_blk) / mk.n,
        "losers_in_closure_blocking": losers <= R_blk,
        "losers_in_closure_all": losers <= R_all,
        "utility_share_closure_blocking": float(share),
        "in_class": in_class,
        "min_margin_blocking": None if mu is None else float(mu),
        "pieces": len(pieces),
        "cover_mechanism_calls": 2 * len(pieces),
        "cover_stability_checks": checks[0],
        "sweep_mechanism_calls": len(us),
        "bound_mean": float(B),
        "bound_valid": valid,
        "bound_over_realized": (float(B / max(L)) if max(L) > 0 else None),
        "qualifying": qualifying,
        "informative": informative,
    }


def ols_r2(Xtr, ytr, Xte, yte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.c_[np.ones(len(Xte)), Xte] @ coef
    ss = ((yte - yte.mean()) ** 2).sum()
    return float(1 - ((yte - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--informative-ratio", type=float, default=0.5)
    ap.add_argument("--kappa-frac", type=float, default=0.25,
                    help="C1: closure size at most this fraction of applicants")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="C3: share of original utility inside the closure")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows, feats = [], []
    t0 = time.time()
    for mid in range(args.markets):
        mk, X, Y, group, elig = make_market(rng)
        S_float = X @ Y.T
        w_fit = group_direction(X, group)
        dirs = {"fitted": w_fit, "rotated_0.5": rotate(w_fit, 0.5, rng),
                "random": rotate(w_fit, np.pi / 2, rng)}
        for dname, w in dirs.items():
            for rel in args.levels:
                set_edit(mk, attenuation_edit(X, Y, w, elig, S_float, rel))
                m0 = deferred_acceptance(mk, Fraction(0))
                m1 = deferred_acceptance(mk, Fraction(1))
                for a, b in INTERVALS:
                    r = analyse_interval(mk, m0, a, b, args.informative_ratio,
                                         args.kappa_frac, args.lam)
                    r.update(market=mid, direction=dname, rel=rel)
                    rows.append(r)
                L1 = float(sum(losses(mk, m0, m1)) / mk.n)
                gap_gain = float(group_gap(mk, m0) - group_gap(mk, m1))
                feats.append({
                    "market": mid, "direction": dname, "rel": rel,
                    "L1": L1, "gap_gain": gap_gain,
                    "global_reversals": len(seed_all_pairs(mk, Fraction(1))) / mk.n,
                    "blocking_reversals": len(seed_blocking(mk, m0, Fraction(1))) / mk.n,
                })
        if (mid + 1) % 10 == 0:
            print(f"  {mid + 1} markets done, {time.time() - t0:.0f}s")

    # Summaries
    def med(key, sub):
        v = [r[key] for r in sub if r[key] is not None]
        return float(np.median(v)) if v else None

    summary = {"closure": {}, "h2": {}, "h1": {}}
    for rel in args.levels:
        sub = [r for r in rows if r["rel"] == rel]
        summary["closure"][str(rel)] = {
            "median_closure_all_frac": med("closure_all_frac", sub),
            "median_closure_blocking_frac": med("closure_blocking_frac", sub),
            "share_losers_in_blocking_closure":
                float(np.mean([r["losers_in_closure_blocking"] for r in sub])),
            "share_losers_in_all_closure":
                float(np.mean([r["losers_in_closure_all"] for r in sub])),
            "median_utility_share_blocking": med("utility_share_closure_blocking", sub),
            "all_bounds_valid": all(r["bound_valid"] for r in sub),
        }
        q = [r for r in sub if r["qualifying"]]
        summary["h2"][str(rel)] = {
            "intervals": len(sub),
            "in_class": int(sum(r["in_class"] for r in sub)),
            "qualifying": len(q),
            "coverage": (float(np.mean([r["informative"] for r in q])) if q else None),
            "median_pieces_qualifying": med("pieces", q),
            "median_regions_qualifying": med("regions", q),
            "median_sweep_mechanism_calls": med("sweep_mechanism_calls", q),
            "median_cover_mechanism_calls": med("cover_mechanism_calls", q),
            "median_cover_stability_checks": med("cover_stability_checks", q),
            "median_bound_over_realized": med("bound_over_realized", q),
        }

    # H1-style comparison: markets split 70/30
    mids = np.arange(args.markets)
    rng.shuffle(mids)
    cut = int(0.7 * len(mids))
    tr, te = set(mids[:cut].tolist()), set(mids[cut:].tolist())
    def mat(keys, part):
        sub = [f for f in feats if f["market"] in part]
        return (np.array([[f[k] for k in keys] for f in sub]),
                np.array([f["L1"] for f in sub]))
    for name, keys in {"edit_size": ["rel"],
                       "global": ["rel", "global_reversals"],
                       "local_blocking": ["rel", "blocking_reversals"]}.items():
        Xtr, ytr = mat(keys, tr)
        Xte, yte = mat(keys, te)
        summary["h1"][name] = ols_r2(Xtr, ytr, Xte, yte)

    summary["example"] = example.verify()
    summary["runtime_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "intervals": rows, "features": feats},
                  fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
