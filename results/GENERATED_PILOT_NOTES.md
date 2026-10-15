# Generated-market pilot: individual harm and verified interval bounds

Code for generated-market experiments on individual harm and verified interval bounds.
All allocation computations use exact rational arithmetic, so priority crossings,
stability checks and bounds are exact for the generated instances.

## Requirements

Python 3.9 or later and NumPy. No other dependencies.

## Run

```
python run_pilot.py --markets 40 --seed 0 --out results.json
```

Options: `--levels` (relative edit norms, default 0.05 0.10),
`--kappa-frac` (C1 threshold as a fraction of applicants, default 0.25),
`--lam` (C3 threshold, default 0.5), `--informative-ratio`
(bound must be at most this multiple of U0, default 0.5).

## What each part checks

| Element | Code | Output field |
|---|---|---|
| Deferred acceptance, stability, harm measures (Eq. harm) | `core.py` | used throughout |
| Priority regions and exact sweep | `path.py` | `regions`, `max_mean_loss` |
| All-pairs seed versus blocking-relevant seed, displacement closure, C1-C3 | `closure.py` | `closure_*_frac`, `utility_share_*`, `min_margin_blocking`, `in_class` |
| Empirical localization: losers inside the closure | `run_pilot.py` | `losers_in_closure_*` |
| Witness bounds and piecewise covers (Eq. witness, Eq. piecewise) | `bounds.py` | `pieces`, `bound_mean`, `bound_valid` |
| H2 qualification and coverage (Eq. coverage) | `run_pilot.py` | `summary.h2` |
| H1-style feature comparison on held-out markets | `run_pilot.py` | `summary.h1` |
| Four-applicant construction  | `example.py` | `summary.example` |

## Interpretation limits

* Localization is checked empirically. A share of 1.0 means no counterexample
  was found in the generated markets; it is not a proof.
* Witness candidates are the applicant- and opportunity-proposing DA outputs at
  the start of each piece. The applicant-optimal common stable allocation of
  the shared-stability literature is not computed, so reported bounds are upper
  bounds on what the best single witness would give.
* Markets are synthetic: Gaussian embeddings with a group shift, four eligible
  opportunities per applicant, capacity three, rank-linear utilities. Results
  should be reported as properties of this generator.
