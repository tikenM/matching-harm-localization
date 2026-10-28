# Reproduce the focused allocation pilot

Read `Pilot_Report.md` first. Results are synthetic and exploratory.

From this directory, install the packages in `requirements.txt`, then run:

```bash
python run_pilot.py
python analyze_results.py
python verify_stability_bound.py
```

The scripts overwrite numerical results and the figure in their own directory. They do not regenerate the narrative report automatically. On the supplied environment the main run takes roughly half a minute on CPU. No GPU, external data, or network is used by the experiment.

- `edits.csv`: 5,760 main interventions; row order aligns with the individual-loss array.
- `individual_losses.npz`: `losses` is a 5,760 × 60 array; accompanying market, level, and direction keys identify each row. All applicants are included, even when their loss is zero.
- `heldout_predictions.csv`: 1,728 interventions from the 72 test markets; predictions and outcomes.
- `predictive_metrics.csv`, `metrics_by_topology.csv`: held-out errors.
- `equal_size_groups.csv`, `equal_size_medians_by_topology.csv`: variation across eight equal-norm directions.
- `illustrative_edits.csv`, `illustrative_individual_losses.csv`: selected complete-eligibility market 70 at 1% score norm. This is an extreme illustrative case, not a representative sample.
- `bound_checks.csv`: 320 additional localized edits across 40 four-pool markets.
- `stability_bound_checks.csv`: independent old-assignment stability checks for 2,482 main edits.
- `summary.json`: estimates, cluster-bootstrap intervals, software versions, seed, runtime.
- `split.json`: exact train/test market identifiers.
- `pilot_results.png`: scientific summary figure. “Local” denotes the strengthened local boundary baseline.
- `audit.json`: extra cross-checks and diagnostic counts.

Model names in CSV: `global_rank` is the global rank-change baseline; `simple_rank` is the strengthened LOCAL boundary baseline, including exposed-applicant utilities and one-hop structural features; `rank_plus_graph` adds multi-hop prospective reachability; `rank_plus_replay` adds post-allocation execution traces. These names do not imply that the local baseline contains only ranking distances.

Loss units derive from linear rank utility, with unmatched utility zero. Global score norm includes ineligible score entries in sparse markets. No fairness-gain matching or deployment claim is made.
