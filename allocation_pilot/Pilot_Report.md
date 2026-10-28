# Focused pilot: structured edits, individual harm, and allocation sensitivity

**Status:** executed synthetic experiment; exploratory, not preregistered. No proposal rewrite. All numbers below come from the accompanying code and saved results.

**Main finding:** equal-sized edits can have very different individual consequences. Global ranking-change counts explain little additional variation beyond edit size in this generator. A much stronger, inexpensive local admission-boundary baseline explains substantially more. Prospective propagation features do **not yet demonstrate a reliable additional benefit** over that baseline. Two conditional guarantees are available, but both are elementary starting points rather than evidence of theoretical novelty.

## 1. What was run

The main experiment contains **240 independent markets and 5,760 interventions**: 80 markets each with complete eligibility, overlapping eligibility on a ring, and three disconnected eligibility pools. Every market has 60 applicants, 15 opportunities, capacity three per opportunity, and eight-dimensional synthetic representations. Applicant preferences, capacities, and tie-breaking stay fixed under every edit. Allocation uses applicant-proposing deferred acceptance (DA), with responsive opportunity priorities determined by scores.

Scores are `S = X Yᵀ / √8`. Applicant representations are Gaussian, with one coordinate shifted by a synthetic binary group label; opportunity representations are Gaussian with a common shift on that coordinate. Applicant preferences are generated independently from a common opportunity-quality term plus individual taste. These are generated representations, **not a trained encoder, real admissions data, or the original manuscripts' full experimental pipeline**.

Each market receives eight rank-one attenuation directions: its group mean-difference direction, five directions tilted from it, and two random controls. The intervention is `X′ = X − γ(Xv)vᵀ`, with unit vector `v`. For each direction, γ is selected to make the **full score-matrix Frobenius norm** of the perturbation exactly 1%, 3%, or 10% of the original score norm. Every realized γ is at most one. Within-market equality holds to numerical tolerance (maximum discrepancy below 7 × 10⁻¹⁷).

Thus “equal-sized” means equal score-perturbation norm. It does not mean equal γ, equal representation distortion, or equal fairness improvement. In sparse markets, this norm also includes scores on ineligible pairs; the complete-eligibility results independently establish that the effect is not solely due to this choice. The raw-space projector differs from the manuscripts' pooled-whitening construction. No claim about fairness improvement is tested here.

Applicant utility is a fixed, explicitly ordinal-derived proxy: 1 for the first acceptable opportunity, decreasing linearly to 0.2 for the last; unmatched utility is zero. Positive loss is `hᵢ = max(uᵢ(old assignment) − uᵢ(new assignment), 0)`. Primary outcome: mean positive loss. Secondary outcomes: fraction with loss greater than 0.05 and maximum individual loss. Gains never cancel losses. These normalized units are not measured economic welfare.

## 2. Equal-sized edits: yes, individual losses differ substantially

For each market and edit norm, the range is taken over all eight directions, without selecting only harmful edits.

| Relative score-edit norm | Median range in mean positive loss | Median range in harmed fraction | Median range in maximum individual loss |
|---|---:|---:|---:|
| 1% | 0.00667 | 1.67 percentage points | 0.267 |
| 3% | 0.01667 | 3.33 percentage points | 0.600 |
| 10% | 0.03238 | 6.67 percentage points | 0.600 |

At the 10% norm, 85.8% of markets have a mean-loss range of at least 0.02. This threshold is a descriptive reporting choice, not a preregistered decision rule.

The result persists in fully connected markets: median maximum-individual-loss ranges are 0.500, 0.600, and 0.486 at the three edit norms.

**Illustration, explicitly selected after observing results:** market 70 has the largest mean-loss range among complete-eligibility markets at the 1% norm. Direction 4 causes no positive individual loss. Direction 2, with the same norm, harms four of 60 applicants, produces mean positive loss 0.03190, and causes a maximum individual loss of 0.82857. The full eight-direction comparison and all 60 applicants' losses are included in separate CSV files. This example illustrates the phenomenon; the table above establishes its broader occurrence in this generator.

**Conclusion:** edit norm alone is insufficient to characterize individual consequences. The experiment does not show that these differences persist at matched fairness gain.

## 3. Ranking changes: global counts are weak; local information is strong

Models are evaluated on **72 completely held-out markets** (24 of each topology), after fitting on 168 markets. All 24 interventions from a market remain in one split. The same fixed Extra Trees specification is used throughout: 200 trees, minimum leaf size eight, all features eligible per split, and fixed random seed. There is no hyperparameter search.

Feature sets:

- **Norm only:** relative score-edit norm.
- **Global rank changes:** norm, maximum score change, inversion count and rate among eligible applicants, number of changed opportunity rankings, maximum inversions at an opportunity, number of applicants involved in reversals, eligibility degree, affected-component fraction, and original allocation welfare and unmatched fraction.
- **Local boundary baseline:** the global features plus reversals between an incumbent and an applicant who prefers that opportunity to their original assignment, original priority margins, number of exposed applicants, their original utility, and their one-hop displacement-graph degree. This is a deliberately strong baseline; it includes local structural and utility information, not just a single ranking distance.
- **Local + prospective graph:** the local baseline plus multi-hop reachable applicant counts, depth, and original utility within the reachable set.
- **Local + replay traces:** the local baseline plus changes in DA proposal, rejection, displacement, and execution-depth counts, measured by rerunning edited DA.

| Feature set | Mean-loss MAE ↓ | Mean-loss R² ↑ | Harmed-fraction MAE ↓ | Maximum-loss MAE ↓ |
|---|---:|---:|---:|---:|
| Norm only | 0.008000 | 0.384 | 0.019994 | 0.232152 |
| Global rank changes | 0.007705 | 0.407 | 0.018531 | 0.226596 |
| Local boundary baseline | 0.003037 | 0.824 | 0.008301 | 0.090534 |
| Local + prospective graph | 0.002947 | 0.832 | 0.008276 | 0.088362 |
| Local + replay traces | 0.002612 | 0.871 | 0.006966 | 0.087877 |

Global changes reduce mean-loss MAE by only about 3.7% relative to norm alone. The local baseline reduces it by about 62%. Therefore the useful distinction is **which admission boundaries change**, together with the affected applicants' situation, rather than the total number of ranking reversals.

These models predict summaries of individual losses at the market-edit level. They do not establish applicant-level prediction or a causal explanation. The difference between the global and local models cannot be attributed solely to local ranking changes, because utility and one-hop features also change.

## 4. Propagation: retrospective value; prospective value remains unproven

The prospective graph starts from the original assignment. A directed edge goes from an applicant to an original incumbent at a lower-preferred acceptable opportunity, representing a possible displacement route if the applicant loses their original assignment. Seeds are both participants in a newly reversed admission-boundary comparison. Reachability uses the original assignment and edited priorities; **it never uses the edited assignment**. The graph is a heuristic, not an asserted envelope containing every possible allocation change.

The replay features instead describe the actual computation of edited DA. They are available only after rerunning the mechanism. Execution-depth counts depend on the fixed queue convention and are not a uniquely identified counterfactual causal chain.

Paired 95% percentile intervals below use 2,000 cluster-bootstrap resamples of held-out markets, stratified by topology. Positive MAE reduction favors the added features.

| Outcome | Prospective graph: MAE reduction [95% interval] | Replay traces: MAE reduction [95% interval] |
|---|---:|---:|
| Mean positive loss | 0.000090 [−0.000025, 0.000213] | 0.000425 [0.000222, 0.000619] |
| Harmed fraction | 0.000024 [−0.000209, 0.000246] | 0.001335 [0.000807, 0.001895] |
| Maximum individual loss | 0.002172 [−0.000277, 0.004757] | 0.002657 [−0.000973, 0.006061] |

The prospective graph's mean-loss improvement is about **3.0%**, and every prospective interval includes zero. Its mean-loss MAE gain is only 0.000034 in complete-eligibility markets. The strongest defensible finding is therefore **no reliable incremental prospective benefit established in this pilot**. This does not prove that propagation is irrelevant; it identifies the strong baseline future methods must beat.

Replay traces reduce mean-loss MAE by approximately 14%, with an interval excluding zero; the harmed-fraction result is also positive. Maximum-loss improvement remains uncertain. Once edited DA has been run and utilities are available, losses can be calculated directly, so these replay predictors provide diagnostic evidence rather than a useful substitute for computing loss.

An initial exploratory comparison omitted exposed-applicant utility and one-hop degree from the local baseline and suggested a larger graph benefit. The final comparison adds those features to address unequal information. All reported numerical results use the strengthened baseline. There was no new test-set split after that change, so this remains exploratory evidence rather than a confirmatory test.

## 5. Conditional guarantees: two attainable starting points

### A. The original assignment remains stable under edited priorities

**Assumptions:** fixed applicant preferences, fixed eligibility and capacities, strict deterministic priorities, responsive opportunity choices, and applicant-proposing DA. Utilities must respect those fixed applicant preferences.

If the original assignment remains stable under the edited opportunity priorities, edited DA gives every applicant an outcome at least as preferred as that original assignment. Consequently **every positive individual loss is zero**. This follows directly from DA's applicant-optimality among stable assignments under the edited priorities [1]. It does not require unchanged rankings or unchanged assignments.

The pilot independently checked original-assignment stability for all **2,482** interventions with no newly blocking admission-boundary comparison. All had zero individual loss. Of these, **2,464 changed eligible rankings and 37 changed assignments**. Fourteen assignment-changing examples occurred in complete-eligibility markets, seven on the connected ring, and sixteen in separate pools.

This establishes a nontrivial connected-market condition permitting reassignments. It is a standard-theory corollary, not a new theorem claimed by this pilot. It guarantees applicant welfare only, not opportunity welfare or group fairness. A changed assignment under this condition can improve applicants despite producing no losses.

### B. Separate eligibility pools, with nonzero harm allowed

Let C be all applicants in connected components of the eligibility graph containing at least one changed opportunity ranking over eligible applicants. With preferences, capacities, eligibility, and tie-breaking fixed, DA decomposes over these components. Outside C, every input ordering is unchanged, hence the allocation is unchanged. Because utilities are nonnegative:

`number of harmed applicants ≤ |C|`

`mean positive loss ≤ (1/n) × Σ[i in C] uᵢ(original assignment)`.

This is a deterministic locality bound. It allows arbitrary ranking and assignment changes inside affected components; it is stronger than an unchanged-ranking condition. It does not cap the severity of loss for any particular applicant inside C.

A separate check used **40 markets and 320 localized rank-one edits**, with 64 applicants and 16 opportunities divided into four eligibility pools. Each opportunity has capacity three. Only applicants in one selected pool receive the rank-one representation edit, at γ = 0.25 or 0.75. This localized intervention is a separate experiment and differs from the global projector used in the main run.

All 320 edits changed eligible rankings; **211 changed assignments**. At most 25% of applicants could be affected. Mean observed positive loss was 0.00905, mean upper bound 0.12305, and the largest observed mean loss 0.05521. There were **zero violations**. The simulation is a consistency check on the proof, not evidence that converts an empirical pattern into a theorem.

The bound is conservative: its average is roughly 14 times the average realized loss. It can become vacuous in a connected market where C contains everyone. The open problem is obtaining useful bounds when the original assignment becomes unstable and competition connects many applicants.

## 6. What this means for the proposed research direction

| Question | Pilot verdict | Consequence |
|---|---|---|
| Equal-sized edits, different individual losses? | Yes, including complete-eligibility markets. | Retain individual-harm sensitivity as a research motivation. |
| Do simple ranking measures already suffice? | Global counts perform weakly; a local boundary baseline with utility and one-hop structure explains much more. | Any proposed method must beat the strengthened local baseline, not just edit norm or Kendall-style counts. |
| Does propagation add information? | Replay traces help retrospectively. Multi-hop prospective benefit is small and uncertain. | Do not claim a validated propagation predictor. Test better-defined propagation conditions on independent markets before making this the central contribution. |
| Is a conditional bound attainable? | Yes: stability preservation gives zero applicant harm; eligibility components give a nonzero exposure/loss bound. | Use these as sanity-check baselines. Research novelty must concern conditions beyond these elementary cases. |

**CRITICAL if claimed in a submission:** “Propagation-aware prediction is established.” Correction: report the negative/inconclusive prospective result and require improvement over the strengthened baseline on a newly generated, untouched evaluation set.

**MAJOR:** applicability to learned fairness edits remains untested. Correction: repeat the fixed-mechanism experiment using an actual frozen encoder and the manuscripts' whitening/erasure map, and compare candidate interventions at matched measured fairness gain.

**MAJOR:** neither conditional guarantee establishes a new contribution. Correction: formulate a checkable condition that permits newly blocking comparisons in a connected market, derive a bound under that condition, and compare its tightness with stability and component baselines.

**MAJOR:** the utility proxy and market generator restrict external validity. Correction: predefine alternative preference-consistent cardinal utilities, seat scarcity, preference-score dependence, eligibility structure, and market sizes; hold out complete generator regimes, not only new random seeds. Include proposer-side specificity and opportunity-side effects in the eventual evaluation.

**MODERATE:** uncertainty is conditional on one split, one fitted model per feature set, one generator family, and the exploratory feature revision. Correction: use untouched follow-up markets and a prospectively fixed comparison. Bootstrap intervals here do not account for training-sample or model-selection uncertainty, and are not multiplicity-adjusted.

## 7. Verification and reproducibility

The simulator checks capacities, acceptability, and absence of blocking pairs for every baseline and edited allocation. Seventy small randomized tests compare opposite proposal-queue orders; a deterministic tied-score case checks tie handling. These are useful checks, not a formal proof or independent solver certification. Saved individual-loss arrays are cross-checked against all reported loss summaries. Training/test market separation is checked. Both locality and stability-preservation conclusions are checked directly.

The main run completed on CPU in roughly half a minute; no GPU or model training was needed. Files include the runnable simulator, independent stability-bound verification, analysis script, all intervention-level outcomes, all individual losses, held-out predictions, split identifiers, bound checks, and the figure. The environment versions and exact runtime are in `summary.json`.

[1] D. Gale and L. S. Shapley (1962), *College Admissions and the Stability of Marriage*, The American Mathematical Monthly 69(1), 9–15, Theorem 2 and the college-admissions extension. [Original paper](https://iqua.ece.toronto.edu/baochun/ece1771f/papers/Gale-Shapley-1962.pdf). The zero-harm implication above is our application of the established applicant-optimality result, not a novelty claim.
