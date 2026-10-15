# JoSAA-calibrated market: displacement closure and landing bound

This script runs the closure and landing-bound checks on a market built from published
JoSAA 2024 IIT data. It tests whether the proposition and the bounds hold, and how informative they are,
on a market with the real structure of the IIT round: real candidates and ranks, real seats by category,
and the reserved-seat categories.

## Data

The CSV files are not redistributed here because the source repository states no licence. Obtain them with:

```
git clone https://github.com/Quantum-Codes/JoSAA_2024
cd JoSAA_2024 && git checkout e700328bd1f0d6d0251fadb3147230564190a395
```

The repository compiles public JoSAA and JIC 2024 report data. Files used, from `exported_data/csv/`:

| File | Content | Use |
|---|---|---|
| `allotment.csv` | JIC 2024 record per JEE (Advanced) candidate: CRL rank, category, category rank, state, allotted IIT program | applicants, ranks, categories, group definition |
| `seat_matrix_25.csv` | IIT seats per program and category | capacities |
| `ORCR.csv` | JoSAA 2024 IIT opening and closing ranks | program desirability; calibration target |
| `choices.csv` | number of candidates listing each IIT program | mean list length |

Official sources: https://josaa.nic.in/document-category/or-cr/ and the JIC 2024 report.

## Run

```
python test_small.py                                   # cross-check against brute-force reference code
python josaa_pilot.py --data JoSAA_2024/exported_data/csv --out josaa_pilot_results.json
```

Python 3.9 or later and NumPy. The full run takes about 70 minutes on two CPU cores.

## Model

- **Applicants:** 36,392 candidates with their real CRL rank, category and category rank. Preparatory-course candidates are excluded.
- **Programs and seats:** 277 IIT programs present in both the 2025 seat matrix and the 2024 rank table, with 17,365 seats.
- **Mechanism R:** one pool per program, priority by CRL rank. This is responsive DA.
- **Mechanism V:** a JoSAA-style encoding of vertical reservations. For each listed program the candidate applies to the OPEN seat first and then to the seat of their category. OPEN priority is by CRL rank and category priority by category rank. Each (program, pool) is a virtual program, so the market is responsive and the proposition applies to it.
- **Preferences:** these are not published, so they are generated.
  - Each candidate's utility for a program is the log of the inverse published OPEN closing rank plus sigma times Gumbel noise.
  - Each candidate lists their top 90 programs. 90 is the total listed choices divided by the number of candidates.
  - sigma was chosen from {0.25, 0.5, 1, 2} to best reproduce the published OPEN closing ranks. At sigma = 0.25 the Spearman correlation is 0.95 and the median ratio of simulated to published closing rank is 1.08.
  - The rank correlation is partly built in, because desirability is taken from the same closing ranks. The level match (ratio 1.08) is the more informative check.
- **Edit:** a hypothetical priority adjustment, not a JoSAA rule.
  - The priority score is -log(rank) + gamma * d * g_i, where g_i = 1 for candidates from the 11 states whose IIT allotment rate is below the median state rate (12,574 candidates).
  - d = -log 0.9 or -log 0.8, so at gamma = 1 a group member's rank is scaled by 0.9 or 0.8.
- **Harm:** rank-linear utility over the programs in the candidate's own list. The OPEN and category seats of one program have equal utility.
- **Checks:** for b in {0.5, 1} and gamma in {0.1, ..., b}:
  - every applicant with a loss is inside the closure R(b);
  - no loss exceeds the landing bound;
  - the closure-only and landing-based bounds are reported.

## Results (`josaa_pilot_results.json`)

| Mech. | d | b | Losers at b | Outside closure | Missed by seed alone | Closure (% matched) | Mean bound, closure (% U0) | Mean bound, landing (% U0) | Realized worst mean loss (% U0) | Max bound, landing | Realized max loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.105 | 0.5 | 762 | 0 | 37 | 98.7 | 97.5 | 4.8 | 1.2 | 0.889 | 0.856 |
| R | 0.105 | 1.0 | 1,536 | 0 | 39 | 99.0 | 98.2 | 9.9 | 2.4 | 0.900 | 0.856 |
| R | 0.223 | 0.5 | 1,617 | 0 | 16 | 99.0 | 98.2 | 10.5 | 2.5 | 0.900 | 0.856 |
| R | 0.223 | 1.0 | 3,027 | 0 | 39 | 99.6 | 99.2 | 21.3 | 5.1 | 0.900 | 0.856 |
| V | 0.105 | 0.5 | 861 | 0 | 554 | 98.4 | 97.2 | 4.7 | 1.4 | 0.822 | 0.822 |
| V | 0.105 | 1.0 | 1,507 | 0 | 828 | 99.0 | 98.1 | 9.1 | 2.4 | 0.867 | 0.822 |
| V | 0.223 | 0.5 | 1,603 | 0 | 419 | 99.0 | 98.2 | 9.7 | 2.6 | 0.867 | 0.822 |
| V | 0.223 | 1.0 | 2,960 | 0 | 553 | 99.4 | 98.9 | 19.8 | 4.8 | 0.889 | 0.822 |

U0 = 0.253 is the mean baseline utility, the trivial bound on mean loss. "Missed by seed alone" counts loser
instances, summed over strengths, that the seed does not contain. At full strength the edit raises the
number of matched group members by 168 to 820.

**Findings**

1. No applicant with a loss fell outside the displacement closure, and no loss exceeded the landing bound, in any of the 8 settings and 60 market–strength pairs. This includes the reserved-seat encoding.
2. The seed alone missed losers in every setting, so the propagation step of the closure is needed.
3. The closure alone does not localize harm in this market. It contains 98 to 99.6% of matched applicants, so the closure-only bounds are close to the trivial bound. The cause is long lists (90 programs) in a near-common prestige order. This extends the trend seen in generated markets, where the closure grew with eligibility degree.
4. The landing-based mean-loss bound is 4.7 to 21.3% of U0, about four times the realized worst mean loss.
5. The landing-based maximum-loss bound is 0.82 to 0.90, against a realized maximum of 0.82 to 0.86, and equal to it in one setting.

## Limits

- Preferences are generated, not observed. The results describe this preference model on the real seat and rank structure.
- The model has one round of DA. It omits JoSAA's multiple rounds, freeze, float and slide choices, withdrawals, de-reservation rules, female-only supernumerary seats (merged into the category pools) and PwD seats (merged into the base category). Candidate gender is not in the data.
- The seat matrix is for 2025 and the ranks are for 2024. Only programs present in both are used.
- Strengths are checked on a grid of 10 values, and realized maxima are lower bounds on the supremum over the interval. The proposition itself covers every strength in the interval.
- Arithmetic is floating point. Ranks are distinct within each pool, so ties do not arise.
- `test_small.py` reproduces DA outcomes, seeds and closures of the brute-force reference code exactly on 400 small random markets with the same category structure (32,800 market–strength pairs).
