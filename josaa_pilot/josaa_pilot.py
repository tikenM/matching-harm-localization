"""JoSAA-calibrated pilot for the displacement closure and the landing bound.

Data (read as plain CSV, see README.md for provenance and limits):
  allotment.csv     JIC 2024 candidate records: CRL rank, category, category rank, state, allotted IIT program
  seat_matrix_25.csv  IIT seat matrix by category (OPEN, EWS, SC, ST, OBC-NCL; PwD and female-only rows merged)
  ORCR.csv          JoSAA 2024 IIT opening and closing ranks (final round)
  choices.csv       number of candidates listing each IIT program (used for mean list length)

Model:
  Applicants  real candidates (preparatory-course candidates excluded), with real CRL and category ranks.
  Seats       real seat counts per program and category.
  Mechanism R single pool per program, priority by CRL rank (responsive).
  Mechanism V JoSAA-style virtual programs: for each listed program, the OPEN seat first and then the
              candidate's category seat; OPEN priority by CRL rank, category priority by category rank.
  Preferences NOT published. Generated: utility = log-desirability (from the published OPEN closing
              rank) + sigma * Gumbel noise; each candidate lists the top L programs. sigma is chosen to
              best reproduce published OPEN closing ranks; L = total choices / number of candidates.
  Edit        hypothetical priority adjustment, not a JoSAA rule: score = -log(rank) + gamma * d * g_i,
              with g_i = 1 for candidates from states whose IIT allotment rate is below the median.
  Harm        rank-linear utility over each candidate's own program list; harm is utility lost relative
              to the unedited allocation (program level; OPEN and category seats of one program are equal).
Checks, for b in {0.5, 1} and gamma on a grid in [0, b]:
  every applicant with positive harm lies in the displacement closure R(b) (localization proposition);
  landing bound >= realized harm for every applicant; closure and landing-based mean and max bounds.
"""
import csv, re, math, sys, json, time, bisect, heapq, argparse, statistics
from collections import defaultdict, Counter
import numpy as np

CATS = ['GEN', 'EWS', 'OBC', 'SC', 'ST']
SEATCOL = {'OPEN': (4, 5), 'EWS': (6, 7), 'SC': (8, 9), 'ST': (10, 11), 'OBC': (12, 13)}


def ni(s): return re.sub(r'[^a-z]', '', s.lower().replace('indian institute of technology', 'iit'))
def npg(s): return re.sub(r'[^a-z0-9]', '', s.lower())


def load(datadir):
    rd = lambda f: list(csv.reader(open(f'{datadir}/{f}', encoding='utf-8')))
    seat = [r for r in rd('seat_matrix_25.csv') if r[0].startswith('Indian Institute of Technology')]
    orcr = rd('ORCR.csv'); allot = rd('allotment.csv'); choices = rd('choices.csv')
    cr = {}
    for r in orcr:
        if r[2] == 'OPEN' and r[3] == 'Gender-Neutral':
            try: cr[(ni(r[0]), npg(r[1]))] = int(float(r[5]))
            except ValueError: pass
    cap = defaultdict(lambda: Counter()); names = {}
    for r in seat:
        key = (ni(r[0]), npg(r[1]))
        if key not in cr: continue
        names[key] = (r[0], r[1])
        for c, (a, b) in SEATCOL.items():
            cap[key][c] += int(r[a]) + int(r[b])
    progs = sorted(cap)
    P = {k: t for t, k in enumerate(progs)}
    capm = [dict(cap[k]) for k in progs]
    closing = [cr[k] for k in progs]
    apps = []
    for r in allot:
        c = r[3]
        if c.startswith('PREP'): continue
        base = {'gen': 'GEN', 'CRLPwD': 'GEN', 'OBC': 'OBC', 'OBCPwD': 'OBC', 'SC': 'SC', 'SCPwD': 'SC',
                'ST': 'ST', 'STPwD': 'ST', 'EWS': 'EWS', 'EWSPwD': 'EWS'}[c]
        crl = int(r[1]) if r[1] else None
        crk = int(r[4]) if r[4] else None
        if crl is None and (crk is None or base == 'GEN'): continue
        apps.append(dict(crl=crl, cat=base, crk=crk, state=r[10], allotted=bool(r[5])))
    total_choices = sum(int(r[3]) for r in choices)
    return progs, names, capm, closing, apps, total_choices


def groups_by_state(apps):
    q = Counter(a['state'] for a in apps); al = Counter(a['state'] for a in apps if a['allotted'])
    rate = {s: al[s] / q[s] for s in q if q[s] >= 50}
    med = statistics.median(rate.values())
    low = {s for s, v in rate.items() if v < med}
    return [1 if a['state'] in low else 0 for a in apps], sorted(low), med


def make_prefs(apps, closing, L, sigma, seed):
    rng = np.random.default_rng(seed)
    v = -np.log(np.array(closing, dtype=float))
    prefs = []
    for _ in apps:
        u = v + sigma * rng.gumbel(size=len(v))
        prefs.append(list(np.argsort(-u)[:L]))
    return prefs


def build_market(apps, prefs, capm, mech):
    """Return virtual market: vprog list of (prog, pool), cap, per-applicant list of virtual ids, base score."""
    vid = {}; vcap = []; vprog = []
    def get(p, pool):
        if (p, pool) not in vid:
            vid[(p, pool)] = len(vcap); vprog.append((p, pool))
            vcap.append(sum(capm[p].values()) if mech == 'R' else capm[p].get(pool, 0))
        return vid[(p, pool)]
    lists = []
    for a, pl in zip(apps, prefs):
        L = []
        for p in pl:
            p = int(p)
            if mech == 'R':
                if a['crl'] is None: continue
                L.append(get(p, 'ALL'))
            else:
                if a['crl'] is not None and capm[p].get('OPEN', 0) > 0:
                    L.append(get(p, 'OPEN'))
                if a['cat'] != 'GEN' and a['crk'] is not None and capm[p].get(a['cat'], 0) > 0:
                    L.append(get(p, a['cat']))
        lists.append(L)
    s_open = [(-math.log(a['crl']) if a['crl'] else -math.inf) for a in apps]
    s_cat = [(-math.log(a['crk']) if a['crk'] else -math.inf) for a in apps]
    return vprog, vcap, lists, (s_open, s_cat)


class Market:
    def __init__(self, vprog, vcap, lists, score, grp, d):
        self.vprog, self.cap, self.lists, self.g, self.d = vprog, vcap, lists, grp, d
        self.s_open, self.s_cat = score
        self.n = len(lists); self.m = len(vcap)
        self.vp = [p for p, _ in vprog]; self.isopen = [pool in ('OPEN', 'ALL') for _, pool in vprog]
        # program-level utility: rank-linear over the distinct programs in the applicant's list
        nprog = max(self.vp) + 1 if self.vp else 0
        self.prank = np.full((self.n, nprog), -1, dtype=np.int16); self.nprogs = [0] * self.n
        for i, L in enumerate(lists):
            t = 0
            for j in L:
                p = self.vp[j]
                if self.prank[i, p] < 0: self.prank[i, p] = t; t += 1
            self.nprogs[i] = t

    def key(self, i, j, gam):  # higher is better; ties by lower index
        s = self.s_open[i] if self.isopen[j] else self.s_cat[i]
        return (s + gam * self.d * self.g[i], -i)

    def da(self, gam):
        held = [[] for _ in range(self.m)]; nxt = [0] * self.n; free = list(range(self.n))
        while free:
            i = free.pop(); L = self.lists[i]
            while nxt[i] < len(L):
                j = L[nxt[i]]; nxt[i] += 1
                if self.cap[j] == 0: continue
                heapq.heappush(held[j], (self.key(i, j, gam), i))
                if len(held[j]) > self.cap[j]:
                    _, r = heapq.heappop(held[j])
                    if r != i: free.append(r)
                    else: continue
                break
        M = [-1] * self.n
        for j in range(self.m):
            for _, i in held[j]: M[i] = j
        return M

    def u(self, i, j): return 0.0 if j < 0 else 1 - int(self.prank[i, self.vp[j]]) / self.nprogs[i]

    def harm(self, M0, M):
        return [max(0.0, self.u(i, M0[i]) - self.u(i, M[i])) for i in range(self.n)]

    def closure(self, M0, b):
        """Seed and displacement closure R(b) of the localization proposition (edit D_ij = d * g_i)."""
        inc = defaultdict(list)
        for i, j in enumerate(M0):
            if j >= 0: inc[j].append(i)
        better = defaultdict(list)  # applicants who list j above their baseline seat
        for i, L in enumerate(self.lists):
            lim = L.index(M0[i]) if M0[i] >= 0 else len(L)
            for j in L[:lim]: better[j].append(i)
        seed = set()
        for j, ks in inc.items():
            cand = sorted(self.key(i, j, 0) for i in better[j] if self.g[i] == 1)
            if not cand: continue
            for k in ks:
                if self.g[k] == 1: continue  # only group members can overtake, and only non-members
                k0 = self.key(k, j, 0)
                lo = bisect.bisect_left(cand, (k0[0] - b * self.d, -math.inf))
                hi = bisect.bisect_left(cand, k0)
                for c in cand[max(lo - 1, 0):hi]:
                    if c < k0 and (c[0] + b * self.d, c[1]) > k0:
                        seed.add(k); break
        # propagation: r in R lists j below M0(r); add incumbents of j that r beats at 0 or at b
        by0 = {j: sorted(ks, key=lambda k: self.key(k, j, 0)) for j, ks in inc.items()}
        byb = {j: sorted(ks, key=lambda k: self.key(k, j, b)) for j, ks in inc.items()}
        p0 = defaultdict(int); pb = defaultdict(int)
        R = set(seed); queue = list(seed)
        while queue:
            r = queue.pop(); L = self.lists[r]
            for j in L[L.index(M0[r]) + 1:]:
                if j not in inc: continue
                for arr, ptr, gam in ((by0[j], p0, 0), (byb[j], pb, b)):
                    kr = self.key(r, j, gam)
                    while ptr[j] < len(arr) and self.key(arr[ptr[j]], j, gam) < kr:
                        k = arr[ptr[j]]; ptr[j] += 1
                        if k not in R: R.add(k); queue.append(k)
        return seed, R

    def enclosure_landing(self, b):
        """Proposer enclosure P over [0, b] and the landing opportunity of each applicant."""
        f = [0 if L else -1 for L in self.lists]
        prop = defaultdict(set)
        for i, L in enumerate(self.lists):
            if L: prop[L[0]].add(i)
        dirty = set(prop); exhausted_reject = [False] * self.n
        while dirty:
            j = dirty.pop(); ps = list(prop[j]); c = self.cap[j]
            k0 = sorted(self.key(i, j, 0) for i in ps)
            gk = sorted(self.key(i, j, 0) for i in ps if self.g[i] == 1)
            for i in ps:
                L = self.lists[i]
                if L[f[i]] != j: continue
                ki = self.key(i, j, 0)
                above = len(k0) - bisect.bisect_right(k0, ki)
                if self.g[i] == 0:  # group members below i at 0 that pass i at b
                    lo = bisect.bisect_left(gk, (ki[0] - b * self.d, -math.inf))
                    hi = bisect.bisect_left(gk, ki)
                    above += sum(1 for x in gk[max(lo - 1, 0):hi] if x < ki and (x[0] + b * self.d, x[1]) > ki)
                if above >= c:
                    if f[i] + 1 < len(L):
                        f[i] += 1; nj = L[f[i]]; prop[nj].add(i); dirty.add(nj)
                    else:
                        exhausted_reject[i] = True
        return f, exhausted_reject

    def landing_bound(self, M0, b):
        f, ex = self.enclosure_landing(b)
        hb = []
        for i in range(self.n):
            u0 = self.u(i, M0[i])
            if f[i] < 0 or ex[i]: hb.append(u0)
            else: hb.append(max(0.0, u0 - self.u(i, self.lists[i][f[i]])))
        return hb


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def closing_fit(mk, M, closing, apps, vprog):
    worst = defaultdict(int)
    for i, j in enumerate(M):
        if j >= 0 and vprog[j][1] in ('OPEN', 'ALL') and apps[i]['crl']:
            worst[vprog[j][0]] = max(worst[vprog[j][0]], apps[i]['crl'])
    ps = sorted(worst)
    sim = [worst[p] for p in ps]; pub = [closing[p] for p in ps]
    ratio = [s / q for s, q in zip(sim, pub)]
    return spearman(sim, pub), statistics.median(ratio), len(ps)


def run(args):
    t0 = time.time()
    progs, names, capm, closing, apps, total_choices = load(args.data)
    grp, low_states, med = groups_by_state(apps)
    L = min(len(progs), round(total_choices / len(apps)))
    out = dict(n_programs=len(progs), n_applicants=len(apps), list_length=L,
               seats=sum(sum(c.values()) for c in capm), group_size=sum(grp),
               group_states=low_states, median_state_rate=med, calibration=[], experiments=[])
    print(f'programs {len(progs)} applicants {len(apps)} seats {out["seats"]} L {L} group {sum(grp)}', flush=True)
    # calibration of sigma on mechanism V at gamma = 0
    best = None
    for sigma in args.sigmas:
        prefs = make_prefs(apps, closing, L, sigma, args.seed)
        vp, vc, ls, sc = build_market(apps, prefs, capm, 'V')
        mk = Market(vp, vc, ls, sc, grp, 0.0)
        M = mk.da(0)
        rho, ratio, k = closing_fit(mk, M, closing, apps, vp)
        out['calibration'].append(dict(sigma=sigma, spearman=rho, median_ratio_sim_to_published=ratio, programs=k,
                                       matched=sum(1 for x in M if x >= 0)))
        print('calib', out['calibration'][-1], f'{time.time()-t0:.0f}s', flush=True)
        if best is None or rho > best[0]: best = (rho, sigma)
    sigma = best[1]; out['sigma'] = sigma
    prefs = make_prefs(apps, closing, L, sigma, args.seed)
    for mech in args.mechs:
        vp, vc, ls, sc = build_market(apps, prefs, capm, mech)
        for d in args.edits:
            mk = Market(vp, vc, ls, sc, grp, d)
            M0 = mk.da(0)
            U0 = sum(mk.u(i, M0[i]) for i in range(mk.n)) / mk.n
            matched0 = sum(1 for x in M0 if x >= 0)
            grid = [k / args.grid for k in range(1, args.grid + 1)]
            runs = {}
            for gam in grid:
                M = mk.da(gam); h = mk.harm(M0, M); runs[gam] = (M, h)
            for b in args.bs:
                seed, R = mk.closure(M0, b)
                hb = mk.landing_bound(M0, b)
                rec = dict(mechanism=mech, d=d, b=b, n=mk.n, matched0=matched0, U0=U0,
                           seed=len(seed), closure=len(R), closure_frac_matched=len(R) / matched0,
                           max_bound_closure=max((mk.u(i, M0[i]) for i in R), default=0.0),
                           mean_bound_closure=sum(mk.u(i, M0[i]) for i in R) / mk.n,
                           max_bound_landing=max((hb[i] for i in R), default=0.0),
                           mean_bound_landing=sum(hb[i] for i in R) / mk.n, strengths=[])
                for gam in grid:
                    if gam > b + 1e-12: continue
                    M, h = runs[gam]
                    losers = {i for i in range(mk.n) if h[i] > 0}
                    gin = [i for i in range(mk.n) if grp[i]]; gout = [i for i in range(mk.n) if not grp[i]]
                    ug = lambda S, MM: sum(mk.u(i, MM[i]) for i in S) / len(S)
                    rec['strengths'].append(dict(
                        gamma=gam, losers=len(losers), losers_outside_closure=len(losers - R),
                        losers_outside_seed=len(losers - seed),
                        landing_violations=sum(1 for i in range(mk.n) if h[i] > hb[i] + 1e-12),
                        max_loss=max(h), mean_loss=sum(h) / mk.n,
                        group_matched_change=sum(1 for i in gin if M[i] >= 0) - sum(1 for i in gin if M0[i] >= 0),
                        gap_before=ug(gout, M0) - ug(gin, M0), gap_after=ug(gout, M) - ug(gin, M)))
                out['experiments'].append(rec)
                last = rec['strengths'][-1]
                print(f"{mech} d={d} b={b}: closure {len(R)} ({len(R)/matched0:.1%} of matched), seed {len(seed)}, "
                      f"losers@b {last['losers']}, outside closure (all strengths) "
                      f"{sum(s['losers_outside_closure'] for s in rec['strengths'])}, landing violations "
                      f"{sum(s['landing_violations'] for s in rec['strengths'])}, {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='josaa_pilot_results.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument('--edits', type=float, nargs='+', default=[-math.log(0.9), -math.log(0.8)])
    ap.add_argument('--bs', type=float, nargs='+', default=[0.5, 1.0])
    ap.add_argument('--grid', type=int, default=10)
    ap.add_argument('--mechs', nargs='+', default=['R', 'V'])
    run(ap.parse_args())
