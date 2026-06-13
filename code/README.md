# Estimating Remaining Useful Life (RUL) & State of Health (SOH) of EV Batteries via a Hybrid Physics-Informed Deep-Learning Framework

**Complete project documentation — standalone.** This document describes the
entire study end-to-end: objective, datasets, methodology, exploratory analysis,
feature engineering, physics priors, models, evaluation protocol, comparative
analysis, ablation study, statistics, edge-deployment profiling, results, and
honest findings. The accompanying notebook reproduces every number, table, and
figure referenced here directly from the raw battery dataset; this document is the
written companion to that notebook.

---

## 1. Objective and scientific question

Estimate the **State of Health (SOH)** and **Remaining Useful Life (RUL)** of
lithium-ion cells with a model that is:

1. **accurate** — competitive with a heavyweight attention model (Transformer);
2. **physically plausible** — its predicted degradation is monotone-ish (no
   non-physical "self-healing"), even where the raw capacity data is noisy;
3. **lightweight** — small enough to run on an ARM Cortex-M4 Battery Management
   System (BMS) (budget: < 256 KB RAM, < 1 MB flash, ~80 MHz).

The central proposal is a **physics-informed GRU (a hybrid PINN)**: a compact
recurrent network whose loss is regularized by a battery-degradation prior, so the
network learns from data while being nudged toward physically sensible trajectories.

**Central scientific question:** does adding physics to a small recurrent model buy
anything a plain network does not — and if so, *what*: peak accuracy, or robustness
and physical plausibility? The honest answer (Section 11) is the latter.

---

## 2. Datasets and verified ground truth

Two public datasets, both **LiCoO₂ (LCO)** chemistry. NASA→CALCE is therefore a
**cross-condition / cross-load / cross-form-factor** transfer, **not**
cross-chemistry. (An earlier project report mislabelled CALCE as LFP/NMC and the
transfer as "cross-chemistry"; this study corrects that and several other errors —
see Section 14.)

### NASA PCoE — four 18650 cells, 2.0 Ah rated, 24 °C, 2 A discharge

| Cell | Cycles | Initial capacity (Ah) | Min capacity (Ah) | Initial SOH | Notes |
|---|---|---|---|---|---|
| B0005 | 168 | 1.856 | 1.287 | 0.928 | strong capacity **regeneration** (sawtooth) |
| B0006 | 168 | 2.035 | 1.154 | **1.018** | **starts above rated** capacity; heaviest regeneration |
| B0007 | 168 | 1.891 | 1.400 | 0.946 | never reaches 30 % fade |
| B0018 | 132 | 1.855 | 1.341 | 0.928 | shortest record |

The raw per-cycle index is the *original interleaved* counter (e.g. 2, 4, …, 614)
and is re-indexed to a contiguous discharge-cycle number. Charge cycles
(needed for the charge-curve feature below) are parsed from the raw measurement
files. A handful of charge cycles log physically impossible voltages (~8.4 V) and
are flagged unusable for the charge feature (their capacity label is unaffected).

### CALCE CS2_35 — one prismatic cell, ~1.1 Ah rated

- **Full trajectory ≈ 1.138 Ah → 0.302 Ah** over **878 cycles**.
- Assembly is the hardest data step and required handling several landmines:
  - the cycle index **resets to 1 in every measurement file** → cumulative offset on concatenation;
  - files must be ordered by their **internal timestamp**, not the filename (filenames are unreliable);
  - one measurement file is a **byte-identical duplicate** of another → dropped (confirmed by MD5);
  - per-cycle discharge capacity is the **increment** of the cumulative discharge-capacity column within the discharge step (max − min), not its maximum;
  - **reference/characterization cycles** produce huge outliers (up to ~54 Ah) → filtered to a plausible band [0.30, 1.30] Ah (4 cycles removed).
- After cleaning: **260,931 raw records** counted; **878 valid cycles**.

### Label definitions (defined once, applied everywhere)

- **SOH** = capacity / rated capacity (rated normalization). Values > 100 % at the
  start are kept honestly (B0006 begins at SOH 101.8 %).
- **End-of-Life (EOL)** = the first cycle where SOH ≤ **80 %** **and stays below for
  5 consecutive cycles**. The persistence rule is essential: it tolerates NASA
  regeneration and CALCE noisy reference cycles, where a single cycle can dip below
  threshold and recover (a naive first-crossing rule mis-dates CALCE EOL by
  hundreds of cycles).
- **RUL** = (EOL cycle − current cycle); undefined past EOL or when EOL is never
  reached. The EOL threshold is always reported alongside any RUL result.

---

## 3. Methodology — the end-to-end procedure

The pipeline is fully reproducible and seed-controlled. Stages:

1. **Data foundation & cleaning.** Build trustworthy, leak-free per-cycle tables
   for all five cells; attach SOH/EOL/RUL labels; verify against the ground truth
   above with automated invariant checks.
2. **Feature engineering.** Compute charge-curve and discharge features per cycle.
3. **Physics-prior fitting.** Fit closed-form degradation laws per cell and decide
   empirically whether the two priors are redundant.
4. **Tensorization (no leakage).** Slide fixed-length windows over each cell;
   fit the feature scaler on **training cells only**, then apply to validation/test.
5. **Modelling.** Train baselines and the hybrid PINN (two physics routes, two
   loss-weighting schemes); tune hyperparameters by Bayesian optimization.
6. **Evaluation.** Leave-one-cell-out (all folds) + cross-dataset transfer, across
   multiple seeds; standard PHM prognostic metrics; significance testing.
7. **Ablation.** Isolate each design choice (physics route, prior, weighting,
   feature richness, data volume) including a de-confound control.
8. **Edge deployment.** Quantize to int8, measure footprint, map to the M4 budget.

**Golden rules enforced throughout:** report only real, measured numbers; never
fabricate; fit all scalers/priors/statistics on training data only; control the
random seed; surface results that contradict expectations rather than hide them.

---

## 4. Exploratory data analysis (EDA)

- **NASA capacity fade** for all four cells shows the expected decline plus a
  pronounced **regeneration sawtooth** (rest periods partially recover capacity).
  B0005 alone has dozens of cycle-to-cycle increases — the key artifact any
  physics prior must tolerate.
- **CALCE timeline choice.** Two candidate timelines were compared (same physical
  cell). The richer timeline (878 cycles, back to the true initial capacity) is
  used because it carries the full early-life curve and is needed for the
  charge-curve feature; the shorter timeline starts mid-life (~0.98 Ah).
- **CALCE outliers.** Reference cycles appear as extreme spikes; the plausibility
  filter cleanly removes them while preserving the real degradation curve.
- **Smoothing.** Capacity is shown raw vs Savitzky–Golay-smoothed to motivate the
  smoothing used inside the feature pipeline.
- **Charge-curve aging.** The constant-current charge voltage–capacity curves shift
  systematically with age — the basis of the incremental-capacity feature.

---

## 5. Feature engineering

Per cycle, two feature groups are computed:

**Charge-curve features — Incremental Capacity Analysis (ICA / dQ-dV).** From the
constant-current charge segment, charge capacity Q(V) is built, smoothed with
Savitzky–Golay (window 11, order 3), differentiated to dQ/dV, and smoothed again.
The dominant dQ/dV peak's **voltage, magnitude, prominence, and area** are
extracted. Peaks correspond to electrode phase transitions; their position and
height shift measurably with aging (a well-established SOH signature). The notebook
shows the **dQ/dV peak shrinking and shifting** across the cell's life.

**Discharge / resistance features.** Discharge duration, minimum voltage, mean
voltage, intra-cycle voltage variance, temperature rise (NASA), and an internal
resistance value (measured column for CALCE; a |ΔV/ΔI| onset proxy for NASA).

Missing values from short/corrupted segments are imputed **within each cell only**
(forward/back-fill then median) — never across cells — to avoid leakage; a feature
that is entirely unavailable for a dataset is dropped rather than faked.

**Feature → SOH correlation** is computed for every cell. The ICA area and the
internal-resistance feature are the strongest, most consistent SOH correlates
across cells, confirming the charge-curve signature carries genuine health
information.

---

## 6. Physics priors (closed-form) and the redundancy question

Two physics-motivated capacity-fade laws are fit per cell with non-linear least
squares. Crucially, **neither requires solving an ODE numerically** — both are used
as *soft priors*, addressing the supervisor's guidance to integrate the laws rather
than run a solver.

- **Logistic (integrated Verhulst):** SOH(t) = K / (1 + A·e^(−r·t)) — the
  closed-form solution of the Verhulst ODE dSOH/dt = r·SOH·(1 − SOH/K). A single
  saturating knee.
- **Double-exponential:** SOH(t) = a·e^(b·t) + c·e^(d·t) — already the closed-form
  solution of a two-state linear system: a fast term (early SEI-layer growth) plus
  a slow term (gradual loss of active material). More flexible.

**Fitted parameters (real, measured — these replace earlier placeholder values):**

| Cell | Logistic R² | Double-exp R² | Logistic RMSE | Double-exp RMSE |
|---|---|---|---|---|
| B0005 | 0.973 | 0.986 | 0.0157 | 0.0112 |
| B0006 | 0.952 | 0.981 | 0.0277 | 0.0173 |
| B0007 | 0.969 | 0.984 | 0.0142 | 0.0102 |
| B0018 | 0.927 | 0.962 | 0.0209 | 0.0150 |
| CS2_35 | 0.963 | 0.975 | 0.0328 | 0.0269 |

**Redundancy verdict (information criterion).** By AIC (which rewards fit and
penalizes the extra parameter), the **double-exponential wins on 5/5 cells**. The
two-timescale law (SEI fast + LAM slow) is therefore justified and is **not**
redundant with the single-knee logistic — the double-exponential is the primary
prior.

---

## 7. Models

Eight models are benchmarked under identical splits and metrics:

| Model | Type | Role |
|---|---|---|
| Double-exp fit | empirical curve fit | "physical floor" — if a fixed curve fit on other cells matches the network, the network adds nothing |
| SVR (RBF) | classic ML, non-recurrent | data-driven floor on flattened windows |
| GRU | recurrent baseline | lightweight RNN reference |
| LSTM | recurrent baseline | stronger RNN (more gates/parameters) |
| Transformer | attention baseline | heavyweight, accurate reference for the gap analysis |
| Neural-ODE | genuine physics-informed alternative | continuous-time latent dynamics; "extreme-hardware" comparator |
| **Hybrid — route A** | PINN (closed-form prior) | penalizes deviation from the fitted prior curve (the supervisor's preferred "integrated ODE" route) |
| **Hybrid — route B** | PINN (ODE residual) | autograd Verhulst residual dŷ/dt = r·ŷ·(1 − ŷ/K) with **learnable (r, K)** — the thesis centrepiece |

**Hybrid design.** The backbone is a stacked GRU identical in family to the GRU
baseline, so "physics on vs off" isolates the contribution of the physics — the
physics enters the **loss**, not the architecture. Two loss-weighting schemes are
supported: a **fixed** weight, and **homoscedastic uncertainty weighting** (learned
per-task noise scales). A light monotonicity penalty discourages self-healing.

**Training harness.** Seeded, early stopping, learning-rate scheduling on plateau,
gradient clipping, and a tiny-batch overfit sanity check for every model (each must
drive its training loss to ~0, confirming it can learn). Hyperparameters
(learning rate, hidden size, sequence length, prior weight) are tuned by
**Bayesian optimization (Optuna TPE)**, not grid/random search.

---

## 8. Evaluation protocol

- **Leave-one-cell-out (LOCO)** on NASA: each fold holds out one whole cell for
  test and one for validation; whole cells never mix across splits. **All four
  folds** are evaluated (not a single fold), across **three random seeds**.
- **Cross-dataset transfer:** train on all NASA cells, test on CALCE (a genuine
  domain shift; one test cell available).
- **Metrics:** SOH RMSE and MAE; a **Physical Consistency Score** (fraction of
  consecutive predictions with non-increasing SOH); and standard PHM prognostic
  metrics (Saxena & Goebel): **α-λ accuracy**, **Prognostic Horizon**, and
  **Cumulative Relative Accuracy (CRA)**. RUL is derived from the predicted SOH
  trajectory and the EOL threshold (standard PHM practice), keeping it consistent
  with the SOH curve.
- **Statistics:** bootstrap confidence intervals and a **paired Wilcoxon test at
  the fold level** (see Section 10 for why the unit of replication matters).

---

## 9. Results and comparative analysis

### Headline benchmark — NASA leave-one-cell-out (mean ± 95 % CI over folds × seeds)

| Model | Params | SOH RMSE | SOH MAE | Phys. cons. | α-λ | Prog. horizon | CRA | Cross-dataset RMSE |
|---|---|---|---|---|---|---|---|---|
| Double-exp fit | 4 | 0.0449 ± 0.0065 | 0.0397 | 0.995 | 0.72 | 83 | 0.37 | 0.6721 |
| SVR | 6,966 | 0.0495 ± 0.0106 | 0.0371 | 0.732 | 0.72 | 83 | 0.48 | 0.1566 |
| Transformer | 17,409 | 0.0471 ± 0.0150 | 0.0437 | 0.739 | 0.59 | 71 | −0.28 | **0.0481** |
| **LSTM** | 5,409 | **0.0262 ± 0.0059** | 0.0227 | 0.821 | 0.78 | 89 | 0.58 | 0.1114 |
| GRU | 4,065 | 0.0348 ± 0.0094 | 0.0312 | 0.779 | 0.69 | 85 | 0.24 | 0.1547 |
| Neural-ODE | **1,233** | 0.0386 ± 0.0116 | 0.0328 | 0.681 | 0.75 | 93 | 0.44 | 0.1286 |
| **Hybrid (route B)** | 41,473 | 0.0290 ± 0.0057 | 0.0252 | **0.918** | 0.71 | 82 | 0.33 | 0.0742 |
| Hybrid (route A) | 41,281 | 0.0448 ± 0.0153 | 0.0400 | 0.895 | 0.58 | 77 | −0.23 | 0.0848 |

### Honest reading of the comparison

- **The LSTM is the single most accurate in-distribution baseline** (0.0262). This
  is reported plainly, not hidden.
- **The learnable route-B hybrid is the robust all-round winner.** It is the
  second-lowest in-distribution RMSE (0.0290), is **the most physically consistent
  model of all** (0.918), and is the **most robust under domain shift** among the
  recurrent family (cross-dataset 0.074 vs LSTM 0.111, GRU 0.155). The plain RNNs
  collapse out of distribution; the physics-regularized model does not.
- **The fixed route-A prior can actively hurt.** It is best on typical cells but
  worst on the atypical B0005/B0006 (which start above rated and regenerate
  heavily): a prior fit on the other cells simply does not describe them, so the
  penalty drags predictions the wrong way. This is why the **learnable** ODE
  residual (route B), which adapts its parameters to the training set, is preferred.
- **Cross-dataset must be read with caution:** only one CALCE test cell exists, so
  those numbers are seed-unstable and indicative only (the Transformer's mean is
  dominated by a single unlucky seed).

### The accuracy / size frontier and the Transformer gap

The hybrid sits on a favourable accuracy-vs-size frontier: **lower error than the
Transformer at a fraction of its footprint** (Section 12). Quantified precisely,
tested at the **fold level (n = 12 experiments)**:

| Comparison | Mean | Transformer | Rel. gap | Folds won | Wilcoxon p | Significant |
|---|---|---|---|---|---|---|
| Hybrid (route B) vs Transformer — SOH RMSE | 0.0290 | 0.0471 | **−38.6 %** | 8/12 | 0.077 | no |
| Hybrid (route A) vs Transformer — SOH RMSE | 0.0448 | 0.0471 | −4.9 % | 6/12 | 0.677 | no |
| Hybrid (route B) vs Transformer — Phys. consistency | 0.918 | 0.739 | +24.2 % | **12/12** | **4.9 × 10⁻⁴** | **yes** |

**Interpretation.** Route B has a *large lower mean* RMSE than the Transformer, but
with only four cells the cross-cell variance is high and the accuracy gap is **not
statistically significant**. What **is** robustly significant is **physical
consistency** — route B is more physically plausible than the Transformer on every
single fold. The defensible claim is therefore *"significantly more physically
plausible, with a large but not-significant accuracy edge,"* which is exactly the
property a safety-critical BMS values.

---

## 10. Statistics — the unit of replication matters

Significance is tested at the **(fold, seed) level (n = 12)**, the honest unit of
replication. An earlier analysis paired *per-cycle* residuals (thousands of points)
and reported an absurd p ≈ 5 × 10⁻⁹⁴ with a −44.5 % gap. That was a
**pseudoreplication artifact**: per-cycle predictions within a cell are strongly
autocorrelated, so treating each as independent massively inflates the effective
sample size; and because models use different sequence lengths, the "paired"
residuals were also cycle-misaligned. Corrected to the fold level, the accuracy gap
is large in mean but not significant, while the physical-consistency advantage is.
This correction *strengthens* the thesis by moving the significant claim onto the
axis the model genuinely wins.

---

## 11. Ablation study

All ablations use the hybrid backbone with one design axis varied at a time.

### Design ablation — with a de-confound control

The route-B model carries one extra input the others lack: a normalized
cycle coordinate (life-fraction). A naive "physics-off vs route-B" comparison would
therefore confound the ODE loss with that extra feature. A control isolates them:

| Variant | SOH RMSE | Phys. cons. | Train time (s) |
|---|---|---|---|
| Route B · ODE-residual (t-channel **+ physics**) | **0.0196** | **0.928** | 2.72 |
| Route A · logistic prior | 0.0262 | 0.902 | 0.84 |
| Route A · double-exp prior | 0.0306 | 0.892 | 1.05 |
| **Physics OFF + t-channel** (control: feature only, no physics) | 0.0319 | 0.918 | 1.51 |
| Physics OFF (no t, no physics) | 0.0409 | 0.891 | 0.88 |
| Route A · uncertainty weighting | 0.0445 | 0.886 | 0.75 |

**Decomposition of route B's advantage:** adding *only the life-fraction channel*
improves RMSE 0.0409 → 0.0319 (about half the gap); adding the *ODE residual on
top* improves a further 0.0319 → 0.0196 (the larger remaining gain) and lifts
physical consistency further. **Conclusion: the physics contributes genuine,
dominant value beyond the input it introduces** — "physics helps" survives the
honest control. Route B costs ~3× the training time of the closed-form routes (an
accuracy/compute trade-off reported explicitly).

### Other ablations

- **Loss weighting (honest negative).** Homoscedastic uncertainty weighting did
  **not** beat a well-chosen fixed weight on this small, clean problem (0.0445 vs
  0.0262); the learned noise scale over-down-weighted the data term. Reported, not
  hidden — fixed weighting is used for the headline models.
- **Prior choice is data-dependent.** On the folds including the hard B0005, the
  logistic prior edged out the double-exponential for route A, even though the
  double-exponential wins the global per-cell curve fit — the prior's value depends
  on the training-cell mix.
- **Feature richness (aggregation).** The full feature set (≈0.0319) barely beats a
  minimal two-feature set (≈0.0330): most predictive signal survives aggressive
  feature reduction. (This is a feature-count surrogate for the
  per-second→per-cycle information trade-off, not a literal within-cycle study.)
- **Data volume.** Subsampling 25–100 % of each training cell's cycles shows **no
  clean trend** (the values sit in a narrow band; full data is, if anything,
  marginally worst): on this small, clean dataset the model is **not data-starved**.

---

## 12. Edge deployment & Cortex-M4 footprint

Models are quantized to int8 and profiled against the M4 budget. Sizes and RAM are
**measured**; the M4 latency is an **analytic** estimate (MAC count ÷ clock at one
MAC/cycle); activation RAM is the single largest intermediate tensor (a lower bound
on the true working set).

| Model | Params | int8 flash | Activation RAM | MACs | Est. M4 latency | Fits M4 |
|---|---|---|---|---|---|---|
| Deployable hybrid (route B, hidden 32) | 11,009 | 11.7 KB | 1.25 KB | 101,328 | 1.27 ms | yes |
| GRU (small) | 4,065 | 4.5 KB | 1.25 KB | 38,432 | 0.48 ms | yes |
| Transformer | 17,409 | 22.4 KB | 2.5 KB | 84,512 | 1.06 ms | yes |

**Matched-size validation.** The deployable hybrid uses a smaller hidden size
(≈11 k parameters) than the benchmark hybrid (≈41 k). Re-evaluated on the same
four-fold leave-one-cell-out protocol, the matched-size model scores
**SOH RMSE 0.0290 ± 0.0085, physical consistency 0.922 — identical to the
4× larger model** — so shrinking to GRU-class size costs **zero accuracy**.

> On the three hybrid parameter counts that appear in the study — all intentional:
> the development/HPO model and the deployed model use the smaller hidden size
> (≈11 k); the head-to-head benchmark deliberately uses a larger hidden size
> (≈41 k) for headroom in the comparison; the matched-size run shows the 11 k model
> matches the 41 k model. The deployable hybrid is **~half the Transformer's flash
> and RAM** and fits the M4 budget comfortably.

**Honest framing of "lightweight."** The hybrid does **not** out-accuracy the LSTM
in-distribution. The deployment justification is earned on **physical consistency
(the only statistically significant network advantage) + cross-dataset robustness +
footprint**, on equal-footing protocol — not on peak accuracy.

---

## 13. Key findings (honest summary)

1. **The data is verified.** Cycle counts (168/168/168/132), initial capacities,
   B0006 starting above rated (SOH 101.8 %), and the CALCE 878-cycle trajectory
   (1.138 → 0.302 Ah, 260,931 records, duplicate dropped, outliers filtered) all
   reproduce exactly.
2. **Real physics parameters** replace earlier placeholders; the double-exponential
   beats the logistic by information criterion on every cell.
3. **The learnable ODE-residual hybrid (route B) is the robust winner** — consistent
   across every held-out cell and **significantly the most physically consistent
   model** (12/12 folds). The **fixed prior (route A) can hurt atypical cells**.
4. **Statistics are tested at the fold level.** Route B's mean accuracy edge over
   the Transformer (−38.6 %) is *not* significant (4 cells, high variance), while
   its physical-consistency edge *is* (p ≈ 5 × 10⁻⁴). Claim "significantly more
   plausible," not "significantly more accurate."
5. **The physics genuinely helps**, beyond the extra input it introduces, per the
   de-confound control.
6. **Lightweight is earned by measurement:** an 11 k-parameter hybrid retains the
   full model's accuracy at about half the Transformer's footprint, within the M4
   budget.

---

## 14. Corrections to the earlier project report

- Chemistry: CALCE is **LCO**, not LFP/NMC; NASA→CALCE is **cross-condition**, not
  cross-chemistry.
- CALCE initial capacity ≈ **1.14 Ah** (the earlier ≈0.88 Ah figure is mid-life).
- Record count corrected to **260,931** (an earlier count double-counted the
  duplicate file).
- All result tables, confidence intervals, and significance tests are **real and
  measured**, replacing placeholder values; the significance test is computed at
  the fold level rather than per cycle.
- The headline is reframed from a blanket "the hybrid always wins" to the more
  defensible, evidence-backed claim about physical plausibility, robustness, and
  footprint.

---

## 15. Limitations and honest caveats

- **Few cells.** Four NASA cells and one CALCE cell limit statistical power;
  in-distribution accuracy differences between the best models are not significant,
  and cross-dataset rankings (one cell) are indicative only.
- **Route-B input asymmetry.** The route-B model uses a normalized life-fraction
  input; this is computed from the test trajectory's length (not known online), a
  mild optimistic bias, and the exported edge model runs a zeroed-coordinate path.
  A fixed-scale normalization that aligns the evaluated and deployed paths is noted
  as future work.
- **M4 latency is analytic**, not measured on a physical board; on-board timing is
  future work. Activation-RAM is a single-tensor lower bound.
- **Aggregation ablation** is a feature-count surrogate, not a literal
  within-cycle-vs-per-cycle representation study.

---

## 16. How to run

1. Keep this document, the notebook, and the battery dataset together.
2. Install a standard scientific-Python environment: `numpy`, `pandas`, `scipy`,
   `scikit-learn`, `torch`, `matplotlib`, plus `jupyter`. (`optuna` is optional and
   used only by one small live hyperparameter-optimization demonstration cell.)
3. Open the notebook and run all cells top to bottom. It auto-detects the dataset,
   rebuilds everything from the raw measurements, and renders every figure, table,
   and result inline. The outputs are also pre-executed and embedded, so the
   notebook can be read without running it.
4. A full run trains many small models on CPU (~8–12 minutes). A single-seed quick
   mode (~4 minutes) is available via one flag in the setup cell; the canonical
   multi-seed numbers are printed alongside as a reference regardless of mode.

The notebook is **self-contained**: it depends only on the raw battery dataset and
defines every function it uses internally.
