"""Market model, deferred acceptance, stability and harm measures.

Scores are exact rationals (fractions.Fraction), so priority comparisons,
crossing points and stability checks are exact. Ties are broken in favour
of the applicant with the smaller index.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional

UNMATCHED = -1


@dataclass
class Market:
    n: int                      # applicants
    m: int                      # opportunities
    cap: List[int]              # capacity of each opportunity
    elig: List[List[bool]]      # elig[i][j]
    pref: List[List[int]]       # pref[i]: eligible opportunities, best first
    S: List[List[Fraction]]     # original scores S[i][j]
    D: List[List[Fraction]]     # score change at full strength D[i][j]
    group: Optional[List[int]] = None   # protected group label, 0 or 1

    def score(self, i: int, j: int, g: Fraction) -> Fraction:
        return self.S[i][j] + g * self.D[i][j]

    def above(self, i: int, k: int, j: int, g: Fraction) -> bool:
        """True if applicant i has higher priority than k at opportunity j."""
        si, sk = self.score(i, j, g), self.score(k, j, g)
        return si > sk or (si == sk and i < k)

    def rank_in_pref(self, i: int, j: int) -> int:
        return self.pref[i].index(j)

    def prefers(self, i: int, j: int, cur: int) -> bool:
        """True if applicant i strictly prefers j to its assignment cur."""
        if cur == UNMATCHED:
            return True
        return self.rank_in_pref(i, j) < self.rank_in_pref(i, cur)


def deferred_acceptance(mk: Market, g: Fraction) -> List[int]:
    """Applicant-proposing deferred acceptance at edit strength g."""
    nxt = [0] * mk.n
    held = {j: [] for j in range(mk.m)}
    free = list(range(mk.n))[::-1]
    while free:
        i = free.pop()
        if nxt[i] >= len(mk.pref[i]):
            continue
        j = mk.pref[i][nxt[i]]
        nxt[i] += 1
        held[j].append(i)
        held[j].sort(key=lambda k: (-mk.score(k, j, g), k))
        if len(held[j]) > mk.cap[j]:
            free.append(held[j].pop())
    match = [UNMATCHED] * mk.n
    for j, lst in held.items():
        for i in lst:
            match[i] = j
    return match


def opportunity_proposing_da(mk: Market, g: Fraction) -> List[int]:
    """Opportunity-proposing deferred acceptance (opportunity-optimal)."""
    order = {}
    for j in range(mk.m):
        cand = [i for i in range(mk.n) if mk.elig[i][j]]
        cand.sort(key=lambda k: (-mk.score(k, j, g), k))
        order[j] = cand
    ptr = [0] * mk.m
    match = [UNMATCHED] * mk.n
    holders = {j: set() for j in range(mk.m)}
    active = True
    while active:
        active = False
        for j in range(mk.m):
            while len(holders[j]) < mk.cap[j] and ptr[j] < len(order[j]):
                i = order[j][ptr[j]]
                ptr[j] += 1
                cur = match[i]
                if mk.prefers(i, j, cur):
                    if cur != UNMATCHED:
                        holders[cur].discard(i)
                    match[i] = j
                    holders[j].add(i)
                    active = True
    return match


def is_stable(mk: Market, match: List[int], g: Fraction) -> bool:
    """Stability of a feasible allocation under priorities at strength g."""
    occupants = {j: [] for j in range(mk.m)}
    for i, j in enumerate(match):
        if j != UNMATCHED:
            if not mk.elig[i][j]:
                return False
            occupants[j].append(i)
    for j in range(mk.m):
        if len(occupants[j]) > mk.cap[j]:
            return False
    for i in range(mk.n):
        for j in mk.pref[i]:
            if not mk.prefers(i, j, match[i]):
                break
            occ = occupants[j]
            if len(occ) < mk.cap[j]:
                return False
            if any(mk.above(i, k, j, g) for k in occ):
                return False
    return True


def utility(mk: Market, i: int, j: int) -> Fraction:
    """Rank-linear utility: 1 for first choice down to 1/5 for the last
    listed choice; 0 when unmatched."""
    if j == UNMATCHED:
        return Fraction(0)
    L = len(mk.pref[i])
    r = mk.rank_in_pref(i, j)
    if L == 1:
        return Fraction(1)
    return Fraction(1) - Fraction(4, 5) * Fraction(r, L - 1)


def losses(mk: Market, m0: List[int], mg: List[int]) -> List[Fraction]:
    return [max(Fraction(0), utility(mk, i, m0[i]) - utility(mk, i, mg[i]))
            for i in range(mk.n)]


def mean_loss(mk: Market, m0, mg) -> Fraction:
    return sum(losses(mk, m0, mg)) / mk.n


def universal_bound(mk: Market, m0) -> Fraction:
    return sum(utility(mk, i, m0[i]) for i in range(mk.n)) / mk.n


def group_gap(mk: Market, match) -> Fraction:
    """Absolute gap between groups in mean applicant utility."""
    if mk.group is None:
        return Fraction(0)
    tot = [Fraction(0), Fraction(0)]
    cnt = [0, 0]
    for i in range(mk.n):
        tot[mk.group[i]] += utility(mk, i, match[i])
        cnt[mk.group[i]] += 1
    if min(cnt) == 0:
        return Fraction(0)
    return abs(tot[0] / cnt[0] - tot[1] / cnt[1])
