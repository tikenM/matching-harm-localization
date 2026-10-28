"""Generate supplementary audit tables from a completed run."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
d=pd.read_csv(P/'edits.csv');e=pd.read_csv(P/'equal_size_groups.csv');p=pd.read_csv(P/'heldout_predictions.csv')
rows=[]
for kind,g in p.groupby('kind'):
    for target in ['mean_positive_loss','hurt_fraction','max_loss']:
        for model in ['norm_only','global_rank','simple_rank','rank_plus_graph','rank_plus_replay']:
            rows.append(dict(kind=kind,target=target,model=model,mae=np.abs(g[target]-g[target+'__'+model]).mean()))
pd.DataFrame(rows).to_csv(P/'metrics_by_topology.csv',index=False)
e.groupby(['kind','level'])[['loss_spread','hurt_spread','maxloss_spread']].median().to_csv(P/'equal_size_medians_by_topology.csv')
case=e[(e.kind=='dense')&(e.level==0)].sort_values(['loss_spread','market_id'],ascending=[False,True]).iloc[0]
mask=(d.market_id==case.market_id)&(d.level==0)
d.loc[mask].to_csv(P/'illustrative_edits.csv',index=False)
z=np.load(P/'individual_losses.npz')
loss=z['losses'][mask.values]
pd.DataFrame(loss.T,columns=['direction_'+str(v) for v in d.loc[mask,'direction']]).rename_axis('applicant_id').to_csv(P/'illustrative_individual_losses.csv')
# Checks against data saved by the runner, independent of simulation assertions.
assert np.allclose(z['losses'].mean(axis=1),d.mean_positive_loss)
assert np.allclose(z['losses'].max(axis=1),d.max_loss)
assert np.allclose((z['losses']>.05).mean(axis=1),d.hurt_fraction)
assert set(p.market_id).isdisjoint(set(json.loads((P/'split.json').read_text())['train']))
summary=dict(illustrative_market_id=int(case.market_id),seedless_edits=int((d.seeds==0).sum()),seedless_changed_edits=int(((d.seeds==0)&(d.assignment_change>0)).sum()),
    graph_saturated_fraction=float((d.reach>=45).mean()),full_norm_min=float(d.relative_norm.min()),full_norm_max=float(d.relative_norm.max()),
    graph_mean_loss_mae_relative_reduction=float((np.abs(p.mean_positive_loss-p.mean_positive_loss__simple_rank).mean()-np.abs(p.mean_positive_loss-p.mean_positive_loss__rank_plus_graph).mean())/np.abs(p.mean_positive_loss-p.mean_positive_loss__simple_rank).mean()))
(P/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
print(pd.DataFrame(rows).query('target=="mean_positive_loss"').to_string(index=False))
