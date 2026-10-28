#!/usr/bin/env python3
"""Reproducible synthetic DA sensitivity pilot. Run: python run_pilot.py"""
from pathlib import Path
from collections import deque
import json, time, platform
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
SEED=20261002

def da(prefs, scores, cap, reverse=False):
    n,m=scores.shape
    ranks=np.empty((n,m),int)
    for j in range(m):
        order=np.lexsort((np.arange(n),-scores[:,j])); ranks[order,j]=np.arange(n)
    held=[[] for _ in range(m)]; match=np.full(n,-1,int); nxt=np.zeros(n,int)
    q=deque(range(n-1,-1,-1) if reverse else range(n)); depth=np.zeros(n,int)
    proposals=rejections=displacements=maxdepth=0
    while q:
        i=q.popleft()
        if nxt[i]>=len(prefs[i]): continue
        j=prefs[i][nxt[i]]; nxt[i]+=1; proposals+=1
        held[j].append(i)
        if len(held[j])<=cap[j]: match[i]=j; continue
        loser=max(held[j],key=lambda k:ranks[k,j]); held[j].remove(loser); rejections+=1
        match[i]=j
        if loser!=i:
            displacements+=1; depth[loser]=depth[i]+1; maxdepth=max(maxdepth,int(depth[loser]))
        match[loser]=-1; q.append(loser)
    return match, np.array([proposals,rejections,displacements,maxdepth],float)

def validate(prefs,s,cap,a):
    n,m=s.shape
    for j in range(m): assert (a==j).sum()<=cap[j]
    for i in range(n):
        assert a[i]==-1 or a[i] in prefs[i]
        for j in prefs[i]:
            if j==a[i]: break
            incumbents=np.where(a==j)[0]
            assert len(incumbents)==cap[j]
            assert all(s[k,j]>s[i,j] or (s[k,j]==s[i,j] and k<i) for k in incumbents)

def market(rng,kind,n=60,m=15,d=8):
    group=rng.integers(0,2,n); x=rng.normal(size=(n,d)); x[:,0]+=.65*(2*group-1)
    y=rng.normal(size=(m,d)); y[:,0]+=.6
    s=x@y.T/np.sqrt(d)
    taste=.8*rng.normal(size=(n,m))+rng.normal(size=m)[None,:]
    prefs=[]; utility=np.zeros((n,m))
    for i in range(n):
        if kind=='dense': eligible=np.arange(m)
        elif kind=='ring': eligible=(np.arange(4)+i%m)%m
        elif kind=='blocks': eligible=np.arange((i//(n//3))*5,(i//(n//3)+1)*5)
        elif kind=='bound': eligible=np.arange((i//(n//4))*4,(i//(n//4)+1)*4)
        p=eligible[np.argsort(-taste[i,eligible],kind='stable')].tolist(); prefs.append(p)
        utility[i,p]=np.linspace(1,.2,len(p))
    return x,y,s,prefs,utility,np.full(m,3),group

def utils(a,u):
    z=np.zeros(len(a)); matched=a>=0; z[matched]=u[np.where(matched)[0],a[matched]]; return z

def components(prefs,m):
    n=len(prefs); adj=[set() for _ in range(n+m)]
    for i,p in enumerate(prefs):
        for j in p: adj[i].add(n+j); adj[n+j].add(i)
    labels=np.full(n+m,-1,int); cid=0
    for v in range(n+m):
        if labels[v]>=0:continue
        q=[v]; labels[v]=cid
        for z in q:
            for w in adj[z]:
                if labels[w]<0:labels[w]=cid;q.append(w)
        cid+=1
    return labels

def features(prefs,s,t,a,u,labels):
    n,m=s.shape; delta=t-s; inv=0; total=0; counts=[]; touched=set(); seeds=set(); direct=0; margins=[]; colleges=[]
    pos=[{j:r for r,j in enumerate(p)} for p in prefs]
    for j in range(m):
        ids=np.array([i for i,p in enumerate(prefs) if j in p]); z1=s[ids,j]; z2=t[ids,j]
        aa,bb=np.triu_indices(len(ids),1); flips=(z1[aa]>z1[bb])!=(z2[aa]>z2[bb]); k=int(flips.sum())
        inv+=k;total+=len(aa);counts.append(k)
        touched.update(ids[aa[flips]].tolist());touched.update(ids[bb[flips]].tolist())
        if k:colleges.append(j)
        for incumbent in np.where(a==j)[0]:
            for i in ids:
                if i==incumbent:continue
                if a[i]==-1 or pos[i][j]<pos[i][a[i]]:
                    margins.append(s[incumbent,j]-s[i,j])
                    if s[incumbent,j]>s[i,j] and t[incumbent,j]<t[i,j]:
                        direct+=1; seeds.add(incumbent);seeds.add(int(i))
    # Hypothetical displacement graph: displaced applicant -> occupants of lower choices.
    # Uses original assignment + edited rankings only; never uses edited assignment.
    adj=[set() for _ in range(n)]
    for i,p in enumerate(prefs):
        lower=p[pos[i][a[i]]+1:] if a[i]>=0 else []
        for j in lower:adj[i].update(np.where(a==j)[0].tolist())
    def reach(hops=None):
        dist={i:0 for i in seeds};q=deque(seeds)
        while q:
            i=q.popleft()
            if hops is not None and dist[i]>=hops:continue
            for k in adj[i]:
                if k not in dist:dist[k]=dist[i]+1;q.append(k)
        return dist
    r=reach();r2=reach(2)
    affected=set(labels[n+j] for j in colleges); incomp=np.array([labels[i] in affected for i in range(n)])
    margins=np.asarray(margins or [0.])
    f=dict(relative_norm=np.linalg.norm(delta)/np.linalg.norm(s),max_abs_delta=np.abs(delta).max(),
        inversion_rate=inv/max(total,1),inversions=inv,changed_colleges=len(colleges),
        max_college_inversions=max(counts),touched_applicants=len(touched),direct_crossings=direct,
        seeds=len(seeds),margin_min=margins.min(),margin_q10=np.quantile(margins,.1),
        margin_median=np.median(margins),baseline_welfare=utils(a,u).mean(),
        baseline_unmatched=(a<0).mean(),eligible_degree=np.mean([len(p) for p in prefs]),
        component_fraction=incomp.mean(),reach=len(r),reach_two=len(r2),
        seed_baseline_utility=utils(a,u)[list(seeds)].sum()/n,
        reach_depth=max(r.values(),default=0),reach_baseline_utility=utils(a,u)[list(r)].sum()/n,
        reach_new=len(set(r)-seeds),seed_outdegree=np.mean([len(adj[i]) for i in seeds]) if seeds else 0)
    return f,incomp

def test_solver():
    rng=np.random.default_rng(123)
    for n in range(3,10):
        for _ in range(10):
            m=3;s=rng.normal(size=(n,m));prefs=[rng.permutation(m)[:rng.integers(1,m+1)].tolist() for i in range(n)];cap=rng.integers(1,3,m)
            a,_=da(prefs,s,cap);b,_=da(prefs,s,cap,True);validate(prefs,s,cap,a);assert np.array_equal(a,b)
    # Deterministic tie handling.
    p=[[0,1]]*5;s=np.zeros((5,2));c=np.array([1,2]);a,_=da(p,s,c);validate(p,s,c,a);assert np.array_equal(a,[0,1,1,-1,-1])

GLOBAL=['relative_norm','max_abs_delta','inversion_rate','inversions','changed_colleges','max_college_inversions','touched_applicants','eligible_degree','component_fraction','baseline_welfare','baseline_unmatched']
BASE=GLOBAL+['direct_crossings','seeds','margin_min','margin_q10','margin_median','seed_baseline_utility','seed_outdegree']
GRAPH=['reach','reach_two','reach_depth','reach_baseline_utility','reach_new']
TRACE=['trace_proposals_delta','trace_rejections_delta','trace_displacements_delta','trace_depth_delta']
TARGETS=['mean_positive_loss','hurt_fraction','max_loss']

def main():
    start=time.time();test_solver();rows=[];individual=[]
    for kind_idx,kind in enumerate(['dense','ring','blocks']):
        for k in range(80):
            mid=kind_idx*80+k;rng=np.random.default_rng(SEED+mid)
            x,y,s,prefs,u,cap,g=market(rng,kind);a,tr=da(prefs,s,cap);validate(prefs,s,cap,a);labels=components(prefs,len(cap))
            gd=x[g==1].mean(0)-x[g==0].mean(0);gd/=np.linalg.norm(gd)
            directions=[gd]
            for angle in [.2,.4,.6,.8,1.0]:
                v=rng.normal(size=x.shape[1]);v-=v@gd*gd;v/=np.linalg.norm(v);directions.append(np.cos(angle)*gd+np.sin(angle)*v)
            for _ in range(2):
                v=rng.normal(size=x.shape[1]);directions.append(v/np.linalg.norm(v))
            ds=[-np.outer(x@v,y@v)/np.sqrt(x.shape[1]) for v in directions]
            norms=np.array([np.linalg.norm(z) for z in ds]);scale=min(1.,norms.min()/(.1*np.linalg.norm(s)))
            for level,rel in enumerate([.01,.03,.1]):
                target=rel*np.linalg.norm(s)*scale
                for direction,delta in enumerate(ds):
                    gamma=target/norms[direction];t=s+gamma*delta;b,tr2=da(prefs,t,cap);validate(prefs,t,cap,b)
                    f,inc=features(prefs,s,t,a,u,labels);loss=np.maximum(utils(a,u)-utils(b,u),0)
                    assert np.all(a[~inc]==b[~inc]);assert gamma<=1+1e-10
                    row=dict(market_id=mid,kind=kind,level=level,direction=direction,gamma=gamma,
                        direction_group_alignment=abs(directions[direction]@gd),
                        representation_relative_norm=np.linalg.norm(gamma*np.outer(x@directions[direction],directions[direction]))/np.linalg.norm(x),
                        mean_positive_loss=loss.mean(),hurt_fraction=(loss>.05).mean(),max_loss=loss.max(),
                        tail_mean_loss=np.sort(loss)[-6:].mean(),assignment_change=(a!=b).mean(),
                        **f,**dict(zip(TRACE,tr2-tr)))
                    rows.append(row)
                    individual.append(loss)
            if k%20==0:print('completed',kind,k,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'edits.csv',index=False)
    np.savez_compressed(OUT/'individual_losses.npz',losses=np.asarray(individual),market_id=df.market_id.values,level=df.level.values,direction=df.direction.values)
    # Within-topology split, market IDs are indivisible; no test-market tuning.
    splitrng=np.random.default_rng(SEED+999);train=[];test=[]
    for ki in range(3):
        ids=splitrng.permutation(np.arange(ki*80,(ki+1)*80));train.extend(ids[:56]);test.extend(ids[56:])
    istrain=df.market_id.isin(train);istest=~istrain
    specs={'norm_only':['relative_norm'],'global_rank':GLOBAL,'simple_rank':BASE,'rank_plus_graph':BASE+GRAPH,'rank_plus_replay':BASE+TRACE}
    pred=df.loc[istest,['market_id','kind','level','direction']+TARGETS].copy();metrics=[];models={}
    for target in TARGETS:
        for name,cols in specs.items():
            model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=8,max_features=1.0,random_state=31415,n_jobs=2)
            model.fit(df.loc[istrain,cols],df.loc[istrain,target]);p=model.predict(df.loc[istest,cols]);pred[target+'__'+name]=p
            metrics.append(dict(target=target,model=name,mae=mean_absolute_error(df.loc[istest,target],p),r2=r2_score(df.loc[istest,target],p)))
    pred.to_csv(OUT/'heldout_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'predictive_metrics.csv',index=False)
    # Paired cluster bootstrap over held-out markets, stratified by topology.
    bootrng=np.random.default_rng(SEED+1000);ci=[]
    for target in TARGETS:
        for added in ['rank_plus_graph','rank_plus_replay']:
            err=np.abs(pred[target]-pred[target+'__simple_rank'])-np.abs(pred[target]-pred[target+'__'+added])
            md=pred.assign(gain=err).groupby(['kind','market_id']).gain.mean()
            samples=[]
            for _ in range(2000):samples.append(np.mean(np.concatenate([bootrng.choice(md.loc[k].values,len(md.loc[k]),replace=True) for k in ['dense','ring','blocks']])))
            ci.append(dict(target=target,comparison=added,mae_reduction=float(err.mean()),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975))))
    # Separate conditional bound check with localized rank-one edits, four disconnected pools.
    bound=[]
    for k in range(40):
        rng=np.random.default_rng(SEED+5000+k);x,y,s,prefs,u,cap,g=market(rng,'bound',64,16);a,tr=da(prefs,s,cap);labels=components(prefs,16)
        v=x[g==1].mean(0)-x[g==0].mean(0);v/=np.linalg.norm(v)
        for pool in range(4):
            mask=np.arange(64)//16==pool
            for gamma in [.25,.75]:
                delta=-gamma*np.outer((x@v)*mask,y@v)/np.sqrt(8);t=s+delta;b,_=da(prefs,t,cap);validate(prefs,t,cap,b)
                f,inc=features(prefs,s,t,a,u,labels);loss=np.maximum(utils(a,u)-utils(b,u),0);B=utils(a,u)[inc].sum()/64
                assert np.all(a[~inc]==b[~inc]);assert loss.mean()<=B+1e-12;assert ((loss>.05).sum()/64)<=inc.mean()+1e-12
                bound.append(dict(market_id=k,pool=pool,gamma=gamma,inversions=f['inversions'],changed_fraction=(a!=b).mean(),hurt_fraction=(loss>.05).mean(),mean_positive_loss=loss.mean(),affected_fraction=inc.mean(),mean_loss_bound=B))
    bd=pd.DataFrame(bound);bd.to_csv(OUT/'bound_checks.csv',index=False)
    # Equal-size comparison, all directions retained, not conditioned on harmful effects.
    eq=df.groupby(['market_id','kind','level']).agg(norm_min=('relative_norm','min'),norm_max=('relative_norm','max'),loss_min=('mean_positive_loss','min'),loss_max=('mean_positive_loss','max'),hurt_min=('hurt_fraction','min'),hurt_max=('hurt_fraction','max'),maxloss_min=('max_loss','min'),maxloss_max=('max_loss','max')).reset_index()
    eq['loss_spread']=eq.loss_max-eq.loss_min;eq['hurt_spread']=eq.hurt_max-eq.hurt_min;eq['maxloss_spread']=eq.maxloss_max-eq.maxloss_min
    eq.to_csv(OUT/'equal_size_groups.csv',index=False)
    summary=dict(seed=SEED,n_markets=240,n_edits=len(df),train_markets=len(train),test_markets=len(test),max_equal_norm_deviation=float((eq.norm_max-eq.norm_min).max()),
        equal_size_by_level=eq.groupby('level')[['loss_spread','hurt_spread','maxloss_spread']].agg(['median','mean','max']).to_json(),
        material_mean_spread_fraction=eq.groupby('level').loss_spread.apply(lambda a:float((a>=.02).mean())).to_dict(),
        material_hurt_spread_fraction=eq.groupby('level').hurt_spread.apply(lambda a:float((a>=.1).mean())).to_dict(),
        metrics=metrics,bootstrap=ci,bound=dict(edits=len(bd),changed_edits=int((bd.changed_fraction>0).sum()),ranking_changed_edits=int((bd.inversions>0).sum()),max_affected_fraction=float(bd.affected_fraction.max()),mean_loss=float(bd.mean_positive_loss.mean()),mean_bound=float(bd.mean_loss_bound.mean()),max_loss=float(bd.mean_positive_loss.max()),violations=0),
        runtime_seconds=time.time()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'split.json').write_text(json.dumps(dict(train=train,test=test),indent=2,default=int))
    # Three compact scientific panels.
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    axes[0].boxplot([eq.loc[eq.level==l,'loss_spread'] for l in range(3)],tick_labels=['1%','3%','10%'],showfliers=False)
    axes[0].set(xlabel='Target relative score-edit norm',ylabel='Within-market range of mean positive loss',title='A. Equal-size edits differ')
    mt=pd.DataFrame(metrics);sub=mt[mt.target=='mean_positive_loss'];axes[1].bar(range(5),sub.mae,color=['#999999','#99bcd5','#477ba5','#2d947c','#b77f43']);axes[1].set_xticks(range(5),['Norm','Global\nrank','Local\nrank','Local +\ngraph','Local +\nreplay']);axes[1].set(ylabel='Held-out mean absolute error',title='B. Prediction of mean positive loss')
    axes[2].scatter(bd.mean_loss_bound,bd.mean_positive_loss,alpha=.25,s=16);axes[2].plot([0,.25],[0,.25],'--',color='gray');axes[2].set(xlabel='Conditional upper bound',ylabel='Observed mean positive loss',title='C. Localized edits in separate pools',xlim=(0,.25),ylim=(0,.25))
    fig.tight_layout();fig.savefig(OUT/'pilot_results.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
