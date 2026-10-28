# Exact comparison-trace audit with an independent round-based check

Exact-arithmetic code for the sweep of applicant-proposing deferred acceptance (DA) along a
linear priority path S + g D, 0 <= g <= 1, and an independent checker of the resulting record.
Python 3.9 or later; standard library only (`fractions`).

| File | Content |
|---|---|
| `trace_sweep.py` | DA that records the comparison behind every rejection; the intervals on which the recorded comparisons stay valid; the sweep of [0,1] into pieces |
| `round_da.py` | Round-based DA (simultaneous proposals), written separately from `trace_sweep.py` |
| `verify_record.py` | Checker: tiling of [0,1], agreement with `round_da` at every crossing strength and between crossings, and validity of each recorded comparison on its piece |
| `cases.py` | Integer rank-one paths and a fitted linear scorer (exact rational least squares), group and random directions |
| `run_audit.py` | Builds cases, sweeps, verifies, applies corruption tests |
| `six_applicant_example.py` | Search for and verification of a six-applicant example |
| `audit_results.json`, `audit_run.log`, `six_applicant_example.json` | Saved outputs |

## Run

```
python run_audit.py --markets 40 --seed 0 --out audit_results.json      # about 30 seconds
python six_applicant_example.py --out six_applicant_example.json
```

## Saved results (`audit_results.json`)

- 160 cases: 40 markets x {integer rank-one, fitted linear scorer} x {group direction, random direction}.
  Markets have 10 to 24 applicants, 3 to 6 opportunities and capacities of 1 to 3.
- All 160 records passed verification.
- In 142 cases the set of matched applicants changes along the path (admission turnover).
  In 145 cases the assignment changes.
- 1,580 pieces in total, at most 48 per case.
- 919 corrupted records (one assignment changed, a boundary shifted, a piece deleted, a closure flag
  flipped, two assignments swapped, a recorded comparison reversed); all 919 were rejected.
- Six-applicant example (seed 12): two opportunities of capacity two, three distinct matched sets,
  four maximal intervals with one assignment (seven trace pieces), and no assignment stable at every
  strength (checked over all 729 assignments).

## Limits

- Generated data. The results concern these generators and sizes.
- The checker shares the tie rule and the input conventions with the sweep. It does not share
  the DA algorithm, the comparison trace or the interval logic. Its crossing set is computed from
  the data, so agreement is checked at every crossing and between crossings, not at a sample.
- Corruption tests cover the six kinds listed. They do not show that every possible
  corruption is rejected.
