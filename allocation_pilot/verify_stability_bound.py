"""Independent check: the original assignment stays stable under edited priorities."""
import numpy as np
import pandas as pd
from run_pilot import OUT, SEED, market, da, validate
D=pd.read_csv(OUT/'edits.csv');checks=[]
for mid,rows in D[D.seeds==0].groupby('market_id'):
    rng=np.random.default_rng(SEED+int(mid));kind=rows.kind.iloc[0]
    x,y,s,prefs,u,cap,g=market(rng,kind);a,_=da(prefs,s,cap)
    gd=x[g==1].mean(0)-x[g==0].mean(0);gd/=np.linalg.norm(gd);directions=[gd]
    for angle in [.2,.4,.6,.8,1.]:
        v=rng.normal(size=x.shape[1]);v-=v@gd*gd;v/=np.linalg.norm(v);directions.append(np.cos(angle)*gd+np.sin(angle)*v)
    for _ in range(2):
        v=rng.normal(size=x.shape[1]);directions.append(v/np.linalg.norm(v))
    for row in rows.itertuples():
        v=directions[row.direction];t=s-row.gamma*np.outer(x@v,y@v)/np.sqrt(x.shape[1])
        validate(prefs,t,cap,a) # Test the OLD assignment, not the new one.
        assert row.max_loss==0
        checks.append(dict(market_id=mid,kind=kind,level=row.level,direction=row.direction,original_assignment_stable=True,inversions=row.inversions,changed_fraction=row.assignment_change,max_loss=row.max_loss))
C=pd.DataFrame(checks);C.to_csv(OUT/'stability_bound_checks.csv',index=False)
print('Independently verified old-assignment stability:',len(C),'edits; changed allocations:',int((C.changed_fraction>0).sum()),'; all individual losses zero.')
