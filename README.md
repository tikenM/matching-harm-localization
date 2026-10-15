# Preliminary code: individual harm from fairness edits in deferred acceptance

Unpublished research code, uploaded for review. It supports a study of which applicants can lose
when a priority input to applicant-proposing deferred acceptance (DA) is edited along an affine path,
and of verified bounds on that loss. Results are from generated markets and from one calibrated market
built on published JoSAA 2024 data. No result here describes a deployed system.

Python 3.9 or later and NumPy (`pip install -r requirements.txt`). No other dependencies.

## Layout

| Path | Content |
|---|---|
| `irg_pilot/` | Exact rational-arithmetic DA, harm measures, priority-path sweep, displacement closure, bounds, market generator |
| `run_pilot.py` | Generated-market pilot (`python run_pilot.py --markets 40 --seed 0 --out results.json`) |
| `checks/localization_check.py` | Every applicant with a loss lies in the displacement closure; the seed alone misses losers (randomized grid check, fixed seeds) |
| `checks/proposer_check.py` | Every DA proposal lies in the proposer enclosure; the landing bound holds |
| `checks/closure_bounds.py` | Tightness of closure, mean-loss, maximum-loss and landing bounds on generated markets |
| `checks/scaling.py` | Closure and enclosure versus eligibility degree (`--part A`, about 10 min) and priority crossings versus market size (`--part B`) |
| `josaa_pilot/josaa_pilot.py`, `test_small.py` | Calibrated market from JoSAA 2024 IIT data; cross-check against brute-force reference code on 400 small markets |
| `results/` | Saved outputs and logs from the runs above |

Run the scripts in `checks/` from the repository root, for example `python checks/closure_bounds.py --markets 30 --seed 0 --out out.json`.


## Data

The JoSAA CSV files are not included because their source states no licence. To obtain them:

```
git clone https://github.com/Quantum-Codes/JoSAA_2024
cd JoSAA_2024 && git checkout e700328bd1f0d6d0251fadb3147230564190a395
python josaa_pilot/josaa_pilot.py --data JoSAA_2024/exported_data/csv --out josaa_out.json
```

Official sources: https://josaa.nic.in/document-category/or-cr/ and the JIC 2024 report.

## Licence
MIT
