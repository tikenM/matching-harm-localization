"""Cross-check josaa_pilot.Market against brute-force reference code on small random markets.

Checks: DA outcome, seed, displacement closure, localization over a fine grid of strengths,
and soundness of the landing bound (realized harm never exceeds it)."""
import random, math, sys
import numpy as np
sys.path.insert(0, '.')
from josaa_pilot import Market, build_market, make_prefs, CATS


def ref_da(n, m, cap, pref, S):
    nxt = [0] * n; held = {j: [] for j in range(m)}; free = list(range(n))
    while free:
        i = free.pop()
        if nxt[i] >= len(pref[i]): continue
        j = pref[i][nxt[i]]; nxt[i] += 1
        held[j].append(i); held[j].sort(key=lambda a: (-S[a][j], a))
        if len(held[j]) > cap[j]:
            free.append(held[j].pop())
    M = [-1] * n
    for j in held:
        for a in held[j]: M[a] = j
    return M


def ref_closure(n, pref, M0, S0, D, b):
    rk = lambda i, j: len(pref[i]) if j < 0 else pref[i].index(j)
    above = lambda g, a, c, j: (S0[a][j] + g * D[a][j], -a) > (S0[c][j] + g * D[c][j], -c)
    seed = set()
    for k in range(n):
        j = M0[k]
        if j < 0: continue
        for i in range(n):
            if i == k or j not in pref[i] or M0[i] == j: continue
            if rk(i, j) < rk(i, M0[i]) and above(0, k, i, j) and above(b, i, k, j): seed.add(k)
    R = set(seed); ch = True
    while ch:
        ch = False
        for k in range(n):
            j = M0[k]
            if k in R or j < 0: continue
            for r in list(R):
                if j in pref[r] and rk(r, M0[r]) < rk(r, j) and (above(0, r, k, j) or above(b, r, k, j)):
                    R.add(k); ch = True; break
    return seed, R


def main(trials=400, seed=7):
    rng = random.Random(seed); stats = dict(da=0, seed=0, closure=0, loc=0, landing=0, checks=0, losers=0)
    for t in range(trials):
        nprog = rng.randint(2, 6); n = rng.randint(8, 40)
        capm = [{c: rng.randint(0, 3) for c in ['OPEN', 'EWS', 'OBC', 'SC', 'ST']} for _ in range(nprog)]
        for c in capm: c['OPEN'] = max(1, c['OPEN'])
        apps = []
        crl = rng.sample(range(1, 10 * n), n)
        for i in range(n):
            cat = rng.choice(CATS); has = rng.random() < 0.8 or cat == 'GEN'
            apps.append(dict(crl=crl[i] if has else None, cat=cat, crk=rng.randint(1, 5 * n), state='x', allotted=False))
        closing = [rng.randint(1, 1000) for _ in range(nprog)]
        prefs = make_prefs(apps, closing, rng.randint(1, nprog), rng.choice([0.5, 2.0]), rng.randint(0, 10**6))
        grp = [rng.randint(0, 1) for _ in range(n)]
        d = rng.choice([0.1, 0.5, 2.0]); b = rng.choice([0.5, 1.0])
        for mech in ('R', 'V'):
            vp, vc, ls, sc = build_market(apps, prefs, capm, mech)
            mk = Market(vp, vc, ls, sc, grp, d)
            m = mk.m
            S0 = [[(mk.key(i, j, 0)[0] if j < m else 0) for j in range(m)] for i in range(n)]
            D = [[d * grp[i]] * m for i in range(n)]
            pref = [[j for j in L if vc[j] > 0] for L in ls]
            M0 = mk.da(0)
            if M0 != ref_da(n, m, vc, pref, S0): stats['da'] += 1
            sd, R = mk.closure(M0, b); rs, rR = ref_closure(n, pref, M0, S0, D, b)
            stats['seed'] += sd != rs; stats['closure'] += R != rR
            hb = mk.landing_bound(M0, b)
            for k in range(41):
                g = b * k / 40
                Sg = [[S0[i][j] + g * D[i][j] for j in range(m)] for i in range(n)]
                M = mk.da(g)
                if M != ref_da(n, m, vc, pref, Sg): stats['da'] += 1
                h = mk.harm(M0, M)
                los = {i for i in range(n) if h[i] > 0}
                stats['losers'] += len(los); stats['checks'] += 1
                stats['loc'] += bool(los - R)
                stats['landing'] += any(h[i] > hb[i] + 1e-12 for i in range(n))
    print(stats)
    return stats


if __name__ == '__main__':
    s = main()
    assert s['da'] == s['seed'] == s['closure'] == s['loc'] == s['landing'] == 0
    print('all checks passed')
