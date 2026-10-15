import random, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from localization_check import da, above
def actual_props(n,m,cap,pref,S):
    nxt=[0]*n; held={j:[] for j in range(m)}; free=list(range(n)); props=set()
    while free:
        i=free.pop()
        if nxt[i]>=len(pref[i]): continue
        j=pref[i][nxt[i]]; nxt[i]+=1; props.add((i,j))
        held[j].append(i); held[j].sort(key=lambda a:(-S[a][j],a))
        if len(held[j])>cap[j]: free.append(held[j].pop())
    return props
def pclosure(n,m,cap,pref,S0,D,a,b):
    P={(i,pref[i][0]) for i in range(n) if pref[i]}
    ch=True
    while ch:
        ch=False
        for (i,j) in list(P):
            A=[k for (k,jj) in P if jj==j and k!=i and (above(S0,D,a,k,i,j) or above(S0,D,b,k,i,j))]
            if len(A)>=cap[j]:
                r=pref[i].index(j)
                if r+1<len(pref[i]) and (i,pref[i][r+1]) not in P:
                    P.add((i,pref[i][r+1])); ch=True
    return P
random.seed(2); fail=0; tot=0; frac=[]
for t in range(3000):
    n=random.randint(3,12); m=random.randint(1,5); cap=[random.randint(1,3) for _ in range(m)]
    elig=[set(random.sample(range(m),random.randint(1,m))) for _ in range(n)]
    pref=[random.sample(sorted(e),len(e)) for e in elig]
    S0=[[random.gauss(0,1) for _ in range(m)] for _ in range(n)]
    D=[[random.gauss(0,1) for _ in range(m)] for _ in range(n)]
    a=random.choice([0,0.25,0.5]); b=a+random.choice([0.1,0.25,0.5])
    P=pclosure(n,m,cap,pref,S0,D,a,b)
    frac.append(len(P)/sum(len(p) for p in pref))
    for g in [a+(b-a)*k/30 for k in range(31)]:
        Sg=[[S0[i][j]+g*D[i][j] for j in range(m)] for i in range(n)]
        tot+=1
        if actual_props(n,m,cap,pref,Sg)-P: fail+=1
import statistics
print('checks',tot,'fail',fail,'median share of eligible pairs enclosed',round(statistics.median(frac),3))

# Landing bound: an applicant ends at an opportunity it proposed to, so its final
# rank is at most the rank of the last enclosed proposal, unless its list is
# exhausted inside P and the last opportunity can reject (then unmatched is possible).
from localization_check import rank
def landing_rank_bound(n, pref, cap, S0, D, P, a, b):
    out = []
    for i in range(n):
        js = [j for j in pref[i] if (i, j) in P]
        if not js:
            out.append(len(pref[i])); continue
        last = js[-1]; r = pref[i].index(last)
        if len(js) == len(pref[i]):
            A = sum(1 for (k, jj) in P if jj == last and k != i
                    and (above(S0, D, a, k, i, last) or above(S0, D, b, k, i, last)))
            if A >= cap[last]:
                r = len(pref[i])
        out.append(r)
    return out
random.seed(3); bad = 0; tot2 = 0
for t in range(3000):
    n = random.randint(3, 12); m = random.randint(1, 5); cap = [random.randint(1, 3) for _ in range(m)]
    elig = [set(random.sample(range(m), random.randint(1, m))) for _ in range(n)]
    pref = [random.sample(sorted(e), len(e)) for e in elig]
    S0 = [[random.gauss(0, 1) for _ in range(m)] for _ in range(n)]
    D = [[random.gauss(0, 1) for _ in range(m)] for _ in range(n)]
    a = random.choice([0, 0.25]); b = a + random.choice([0.1, 0.25, 0.5])
    P = pclosure(n, m, cap, pref, S0, D, a, b)
    lb = landing_rank_bound(n, pref, cap, S0, D, P, a, b)
    for g in [a + (b - a) * k / 30 for k in range(31)]:
        Sg = [[S0[i][j] + g * D[i][j] for j in range(m)] for i in range(n)]
        Mg, _ = da(n, m, cap, elig, pref, Sg)
        tot2 += 1
        if any(rank(pref, i, Mg[i]) > lb[i] for i in range(n)): bad += 1
print('landing-bound checks', tot2, 'violations', bad)
