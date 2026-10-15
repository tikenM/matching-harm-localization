"""Density and scale experiments (float arithmetic; synthetic generator of irg_pilot).
A: closure, proposer enclosure and landing bound versus eligibility degree and edit norm.
B: number of priority crossings and DA time versus market size, for two ways of growing the market."""
import os, sys, time, json, argparse
import numpy as np
sys.path.insert(0, os.environ.get("IRG_PILOT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from irg_pilot.generate import make_market, group_direction, attenuation_edit

def util_table(pref, n, m):
    U = np.zeros((n, m))
    for i in range(n):
        L = len(pref[i])
        for r, j in enumerate(pref[i]):
            U[i, j] = 1.0 if L == 1 else 1 - 0.8 * r / (L - 1)
    return U

def da(pref, cap, S, D, g, n, m):
    sc = S + g * D
    nxt = [0] * n; held = [[] for _ in range(m)]; free = list(range(n))[::-1]
    while free:
        i = free.pop()
        if nxt[i] >= len(pref[i]): continue
        j = pref[i][nxt[i]]; nxt[i] += 1
        held[j].append(i); held[j].sort(key=lambda k: (-sc[k, j], k))
        if len(held[j]) > cap[j]: free.append(held[j].pop())
    M = [-1] * n
    for j in range(m):
        for i in held[j]: M[i] = j
    return M

def above(S, D, g, k, i, j):
    a = S[k, j] + g * D[k, j]; b = S[i, j] + g * D[i, j]
    return a > b or (a == b and k < i)

def closure(pref, S, D, M0, b, n, m, elig_at):
    rank = [{j: r for r, j in enumerate(p)} for p in pref]
    seed = set()
    for k in range(n):
        j = M0[k]
        if j < 0: continue
        for i in elig_at[j]:
            if i == k or M0[i] == j: continue
            if (M0[i] < 0 or rank[i][j] < rank[i][M0[i]]) and above(S, D, 0, k, i, j) and above(S, D, b, i, k, j):
                seed.add(k); break
    R = set(seed); ch = True
    while ch:
        ch = False
        for k in range(n):
            j = M0[k]
            if k in R or j < 0: continue
            for r in elig_at[j]:
                if r in R and M0[r] >= 0 and M0[r] != j and rank[r][M0[r]] < rank[r][j] and (above(S, D, 0, r, k, j) or above(S, D, b, r, k, j)):
                    R.add(k); ch = True; break
    return R

def proposer_enclosure(pref, cap, S, D, b, n, m):
    P = {(i, pref[i][0]) for i in range(n) if pref[i]}
    Pj = [set() for _ in range(m)]
    for i, j in P: Pj[j].add(i)
    work = list(P)
    while work:
        i, j = work.pop()
        c = sum(1 for k in Pj[j] if k != i and (above(S, D, 0, k, i, j) or above(S, D, b, k, i, j)))
        if c >= cap[j]:
            r = pref[i].index(j)
            if r + 1 < len(pref[i]):
                nj = pref[i][r + 1]
                if i not in Pj[nj]:
                    Pj[nj].add(i); P.add((i, nj)); work.append((i, nj))
                    # new proposer to nj can raise counts of others there
                    for k in list(Pj[nj]):
                        if k != i: work.append((k, nj))
    return P, Pj

def landing(pref, cap, S, D, b, M0, U, P, Pj, n):
    out = np.zeros(n)
    for i in range(n):
        js = [j for j in pref[i] if i in Pj[j]]
        u0 = U[i, M0[i]] if M0[i] >= 0 else 0.0
        if not js: out[i] = u0; continue
        last = js[-1]; floor = U[i, last]
        if len(js) == len(pref[i]):
            c = sum(1 for k in Pj[last] if k != i and (above(S, D, 0, k, i, last) or above(S, D, b, k, i, last)))
            if c >= cap[last]: floor = 0.0
        out[i] = max(0.0, u0 - floor)
    return out

def exp_A(markets, n, m, cap, degs, rels, seed):
    rows = []
    for deg in degs:
        for rel in rels:
            rng = np.random.default_rng(seed)
            vals = []
            for _ in range(markets):
                mk, X, Y, group, elig = make_market(rng, n=n, m=m, cap=cap, deg=deg)
                S = X @ Y.T; w = group_direction(X, group)
                D = attenuation_edit(X, Y, w, elig, S, rel)
                pref = mk.pref; capl = mk.cap; el_at = [list(np.where(elig[:, j])[0]) for j in range(m)]
                U = util_table(pref, n, m)
                M0 = da(pref, capl, S, D, 0.0, n, m); u0 = np.array([U[i, M0[i]] if M0[i] >= 0 else 0 for i in range(n)])
                b = 1.0
                R = closure(pref, S, D, M0, b, n, m, el_at)
                P, Pj = proposer_enclosure(pref, capl, S, D, b, n, m)
                lb = landing(pref, capl, S, D, b, M0, U, P, Pj, n)
                real_max = 0.0; real_mean = 0.0; bad = False
                for g in np.linspace(0, 1, 11):
                    Mg = da(pref, capl, S, D, g, n, m)
                    h = np.array([max(0.0, u0[i] - (U[i, Mg[i]] if Mg[i] >= 0 else 0.0)) for i in range(n)])
                    real_max = max(real_max, h.max()); real_mean = max(real_mean, h.mean())
                    if any(h[i] > 1e-12 and i not in R for i in range(n)): bad = True
                    if (h > lb + 1e-12).any(): bad = True
                Rl = list(R)
                vals.append(dict(closure=len(R) / n, P=len(P) / sum(len(p) for p in pref), U0=u0.mean(),
                    bmean=u0[Rl].sum() / n if Rl else 0.0, bmax=u0[Rl].max() if Rl else 0.0,
                    lmax=lb[Rl].max() if Rl else 0.0, lmean=lb[Rl].sum() / n if Rl else 0.0,
                    real_max=real_max, real_mean=real_mean, bad=bad))
            med = lambda k: float(np.median([v[k] for v in vals]))
            rows.append(dict(deg=deg, rel=rel, closure=med('closure'), P=med('P'),
                bmean_over_U0=float(np.median([v['bmean'] / v['U0'] for v in vals])),
                lmean_over_U0=float(np.median([v['lmean'] / v['U0'] for v in vals])),
                bmax=med('bmax'), lmax=med('lmax'), real_max=med('real_max'), real_mean=med('real_mean'),
                any_violation=any(v['bad'] for v in vals)))
            print(rows[-1], flush=True)
    return rows

def exp_B(configs, rel, seed):
    rows = []
    for name, n, m, deg in configs:
        rng = np.random.default_rng(seed)
        cap = int(np.ceil(0.75 * n / m))
        mk, X, Y, group, elig = make_market(rng, n=n, m=m, cap=cap, deg=deg)
        S = X @ Y.T; w = group_direction(X, group)
        D = attenuation_edit(X, Y, w, elig, S, rel)
        pairs = 0; cross = 0
        for j in range(m):
            el = np.where(elig[:, j])[0]
            if len(el) < 2: continue
            s = S[el, j]; d = D[el, j]
            d0 = s[:, None] - s[None, :]; dd = d[:, None] - d[None, :]
            iu = np.triu_indices(len(el), 1)
            d0 = d0[iu]; dd = dd[iu]; pairs += len(d0)
            with np.errstate(divide='ignore', invalid='ignore'):
                root = -d0 / dd
            cross += int(((root > 0) & (root < 1)).sum())
        t = time.time(); da(mk.pref, mk.cap, S, D, 0.5, n, m); td = time.time() - t
        rows.append(dict(config=name, n=n, m=m, cap=cap, deg=deg, rel=rel, eligible_pairs=int(sum(len(p) for p in mk.pref)),
            same_opportunity_pairs=pairs, crossings=cross, da_seconds_python=round(td, 3),
            sweep_hours_python_est=round(cross * td / 3600, 2)))
        print(rows[-1], flush=True)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=8)
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    a = ap.parse_args()
    if a.part in ("A", "both"):
        json.dump(exp_A(a.markets, 200, 40, 3, [4, 8, 16, 32], [0.05, 0.10, 0.20], 0), open("scaling_A.json", "w"), indent=1)
    if a.part in ("B", "both"):
        json.dump(exp_B([("m grows with n", 1000, 250, 8), ("m grows with n", 5000, 1250, 8),
                         ("m fixed", 1000, 100, 8), ("m fixed", 5000, 100, 8), ("m fixed", 20000, 100, 8)], 0.10, 0),
                  open("scaling_B.json", "w"), indent=1)
