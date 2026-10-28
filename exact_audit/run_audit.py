"""Run the exact comparison-trace audit and the independent check.

  python run_audit.py [--markets 40] [--seed 0] [--out audit_results.json]

For each market seed, four cases are built: {integer rank-one, fitted linear scorer} x
{group direction, random direction}. For each case the trace sweep is computed and the
record is verified by verify_record.verify (round-based DA at every crossing and between
crossings, tiling, and certificate checks). Corruption tests then modify each record and
require rejection. Reported: cases, cases with admission turnover (the set of matched
applicants changes along the path), pieces, verification failures, corruption outcomes.
"""
import argparse, copy, json, random, time
from fractions import Fraction
from trace_sweep import Market, sweep
from verify_record import verify
from cases import make_case

ZERO, ONE = Fraction(0), Fraction(1)


def matched_set(match):
    return frozenset(i for i, j in enumerate(match) if j != -1)


def corruptions(rec, rng, n, m):
    """Yield (name, corrupted copy)."""
    k = len(rec)
    p = rng.randrange(k)
    c = copy.deepcopy(rec)
    i = rng.randrange(n)
    c[p]["match"][i] = (c[p]["match"][i] + 2) % (m + 1) - 1 if c[p]["match"][i] != -1 else 0
    if c[p]["match"] != rec[p]["match"]:
        yield "change one assignment", c
    if k >= 2:
        c = copy.deepcopy(rec)
        q = rng.randrange(k - 1)
        d = (c[q]["hi"] - c[q]["lo"]) / 4
        if d > 0:
            c[q]["hi"] -= d
            yield "shift a boundary (gap)", c
        c = copy.deepcopy(rec)
        del c[rng.randrange(k)]
        yield "delete a piece", c
        c = copy.deepcopy(rec)
        q = rng.randrange(k - 1)
        c[q]["hi_closed"] = not c[q]["hi_closed"]
        yield "flip a closure flag", c
        if any(rec[a]["match"] != rec[b]["match"] for a in range(k) for b in range(a)):
            c = copy.deepcopy(rec)
            a, b = next((a, b) for a in range(k) for b in range(a) if rec[a]["match"] != rec[b]["match"])
            c[a]["match"], c[b]["match"] = c[b]["match"], c[a]["match"]
            yield "swap two pieces' assignments", c
    tr = [(q, t) for q in range(k) for t in range(len(rec[q]["trace"]))]
    if tr:
        q, t = rng.choice(tr)
        c = copy.deepcopy(rec)
        j, w, l = c[q]["trace"][t]
        c[q]["trace"][t] = (j, l, w)
        yield "reverse a recorded comparison", c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="audit_results.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    res = dict(cases=0, verified=0, failed=[], turnover_cases=0, changed_assignment_cases=0,
               pieces_total=0, pieces_max=0, by_type={}, corruptions=0, rejected=0, accepted=[])
    skipped = 0
    for s in range(a.seed * 100000, a.seed * 100000 + a.markets):
        for scorer in ("integer", "fitted"):
            for direction in ("group", "random"):
                case = make_case(s * 7 + (scorer == "fitted") * 3 + (direction == "random"), scorer, direction)
                if case is None:
                    skipped += 1
                    continue
                mk = Market(case["pref"], case["cap"], case["S"], case["D"])
                rec = sweep(mk)
                ok, why = verify(case["pref"], case["cap"], case["S"], case["D"], rec)
                res["cases"] += 1
                key = scorer + "/" + direction
                bt = res["by_type"].setdefault(key, dict(cases=0, turnover=0))
                bt["cases"] += 1
                if ok:
                    res["verified"] += 1
                else:
                    res["failed"].append((s, key, why))
                sets = {matched_set(p["match"]) for p in rec}
                changed = len({tuple(p["match"]) for p in rec}) > 1
                res["turnover_cases"] += len(sets) > 1
                bt["turnover"] += len(sets) > 1
                res["changed_assignment_cases"] += changed
                res["pieces_total"] += len(rec)
                res["pieces_max"] = max(res["pieces_max"], len(rec))
                for name, bad in corruptions(rec, rng, case["n"], case["m"]):
                    res["corruptions"] += 1
                    ok2, why2 = verify(case["pref"], case["cap"], case["S"], case["D"], bad)
                    if ok2:
                        res["accepted"].append((s, key, name))
                    else:
                        res["rejected"] += 1
    res["skipped_singular_or_degenerate"] = skipped
    res["runtime_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str))
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
