import random, itertools
def da(n,m,cap,elig,pref,S):
    # applicant-proposing DA; pref[i] list of opps in order; priority by S[i][j] (higher better), tie by lower index
    nxt=[0]*n; held={j:[] for j in range(m)}; free=list(range(n)); rej=[]  # rej: (time,i,j)
    t=0
    while free:
        i=free.pop()
        if nxt[i]>=len(pref[i]): continue
        j=pref[i][nxt[i]]; nxt[i]+=1
        held[j].append(i); held[j].sort(key=lambda a:(-S[a][j],a))
        if len(held[j])>cap[j]:
            r=held[j].pop(); t+=1; rej.append((t,r,j)); free.append(r)
    M=[None]*n
    for j in held:
        for a in held[j]: M[a]=j
    return M,rej
def above(S0,D,g,a,b,j):  # a above b at j at strength g
    va=S0[a][j]+g*D[a][j]; vb=S0[b][j]+g*D[b][j]
    return (va,-a)>(vb,-b)
def rank(pref,i,j):
    return len(pref[i]) if j is None else pref[i].index(j)
def closure(n,m,elig,pref,M0,S0,D,bb,tight):
    seed=set(); 
    for k in range(n):
        j=M0[k]
        if j is None: continue
        for i in range(n):
            if i==k or j not in elig[i] or M0[i]==j: continue
            if rank(pref,i,j)<rank(pref,i,M0[i]) and above(S0,D,0,k,i,j) and above(S0,D,bb,i,k,j):
                seed.add(k)
                if not tight: seed.add(i)
    R=set(seed); ch=True
    while ch:
        ch=False
        for k in range(n):
            j=M0[k]
            if k in R or j is None: continue
            for r in R:
                if j in elig[r] and (above(S0,D,0,r,k,j) or above(S0,D,bb,r,k,j)) and (not tight or rank(pref,r,M0[r])<rank(pref,r,j)):
                    R.add(k); ch=True; break
    return R,seed
if __name__=="__main__":
    random.seed(1); fails={'loose':0,'tight':0,'seedonly':0}; tot=0; sizes=[]
    for trial in range(4000):
        n=random.randint(3,12); m=random.randint(1,5); cap=[random.randint(1,3) for _ in range(m)]
        elig=[set(random.sample(range(m),random.randint(1,m))) for _ in range(n)]
        pref=[random.sample(sorted(elig[i]),len(elig[i])) for i in range(n)]
        S0=[[random.gauss(0,1) for _ in range(m)] for _ in range(n)]
        D=[[random.gauss(0,1)*random.choice([0.3,1,3]) for _ in range(m)] for _ in range(n)]
        M0,_=da(n,m,cap,elig,pref,S0)
        bb=random.choice([0.25,0.5,1.0])
        R1,_=closure(n,m,elig,pref,M0,S0,D,bb,False); R2,seed=closure(n,m,elig,pref,M0,S0,D,bb,True)
        assert R2<=R1
        for g in [bb*k/40 for k in range(41)]:
            Sg=[[S0[i][j]+g*D[i][j] for j in range(m)] for i in range(n)]
            Mg,_=da(n,m,cap,elig,pref,Sg)
            losers={i for i in range(n) if rank(pref,i,Mg[i])>rank(pref,i,M0[i])}
            tot+=1
            if losers-R1: fails['loose']+=1
            if losers-R2: fails['tight']+=1
            if losers-seed: fails['seedonly']+=1
        sizes.append((len(R2),len(R1),n))
    print('checks',tot,fails)
    import statistics; print('median tight/loose frac',statistics.median(a/c for a,b,c in sizes),statistics.median(b/c for a,b,c in sizes))
    