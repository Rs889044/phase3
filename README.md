# Estimating Remaining Useful Life (RUL) & State of Health (SOH) of EV Batteries via a Hybrid Physics-Informed Deep-Learning Framework

**Complete project documentation — standalone.** This document describes the
entire study end-to-end: objective, datasets, methodology, exploratory analysis,
feature engineering, physics priors, models, evaluation protocol, comparative
analysis, ablation study, statistics, edge-deployment profiling, results, and
honest findings. The accompanying notebook reproduces every number, table, and
figure referenced here directly from the raw battery dataset; this document is the
written companion to that notebook.

> **Note (2026-07-10).** The NASA charge↔discharge pairing was corrected
> (the raw `.mat` sequence contains extra/corrupted charge operations that a
> positional pairing silently absorbed, leaving most ICA features 1–2 cycles
> stale) and the full pipeline was re-executed. A later pass the same day
> re-ran the design/aggregation/volume ablations on the full benchmark
> protocol (they were previously 2 folds × 2 seeds); Section 11 reflects the
> full-protocol run, whose control and route rows reproduce the benchmark
> rows exactly. All prose and tables below reflect the post-fix rerun; the
> executed notebook and its `tables/`/`figures/` outputs remain the
> authoritative numbers.

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
anything a plain network does not — and if so, *what*? The measured answer
(Sections 9–11): at matched capacity the physics term delivers a real accuracy
gain in-distribution, plus the best physical plausibility among the learned
models — while peak cross-dataset robustness and prognostic calibration go to
other models, and both negative results are reported with mechanisms.

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
files; each discharge is paired with the **last charge operation preceding it in
the raw sequence** (positional pairing is wrong for these files — see the Note
above). A handful of charge cycles log physically impossible voltages (~8.4 V) and
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
   feature richness, data volume) on the same full four-fold protocol as the
   benchmark; the control and route rows re-run the benchmark configurations
   exactly, tying the two tables together (Section 11).
8. **Edge deployment.** Quantize to int8, measure footprint, map to the M4 budget.

**Golden rules enforced throughout:** report only real, measured numbers; never
fabricate; fit all scalers/priors/statistics on training data only; control the
random seed (every model is seeded immediately before construction, so results
are independent of notebook execution order); surface results that contradict
expectations rather than hide them.

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
  smoothing used inside the feature pipeline (the SOH *label* itself stays raw).
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
voltage, intra-cycle voltage variance, and an internal resistance value
(measured column for CALCE; a |ΔV/ΔI| onset proxy for NASA). A temperature-rise
summary (NASA) and the ICA peak prominence are also computed per cycle, but as
diagnostics — the models receive the eight shared features plus the cycle
coordinate.

Missing values from short/corrupted segments are imputed **within each cell only**
(forward/back-fill then median) — never across cells — to avoid leakage; a feature
that is entirely unavailable for a dataset is dropped rather than faked.

**Feature → SOH correlation** is computed for every cell. The ICA area and the
internal-resistance feature are the strongest, most consistent SOH correlates
across cells, confirming the charge-curve signature carries genuine health
information. One caution is carried through the whole study: under
constant-current lab cycling, **discharge time and ICA area are near-proxies of
the same-cycle capacity from which the label is computed**, so the pointwise SOH
task is easier than a field deployment would be — every competent model lands in
a narrow error band, and the informative differences show up in physical
consistency, robustness, and footprint rather than raw RMSE.

---

## 6. Physics priors (closed-form) and the redundancy question

Two physics-motivated capacity-fade laws are fit per cell with non-linear least
squares. Crucially, **neither requires solving an ODE numerically** — both are used
as *soft priors*, addressing the supervisor's guidance to integrate the laws rather
than run a solver.

- **Logistic (integrated Verhulst):** SOH(t) = K / (1 + A·e^(−r·t)) — the
  closed-form solution of the Verhulst ODE dSOH/dt = r·SOH·(1 − SOH/K). Every fit
  converges to a *negative* rate constant, under which the curve decays from its
  starting value K/(1+A) toward zero — so the fitted K is a scale parameter of
  the decay, not a plateau the trajectory approaches.
- **Double-exponential:** SOH(t) = a·e^(b·t) + c·e^(d·t) — a flexible
  two-exponential form *motivated by* the two-timescale picture (fast SEI
  stabilization + slow loss of active material). The mechanistic reading is not
  imposed on the fit: the unconstrained coefficients do not always take the signs
  of two decay processes (e.g. a negative fast-term amplitude on B0005, a growing
  second term on B0006), so the form is used as a flexible empirical prior, not a
  literal SEI/LAM decomposition.

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
more flexible four-parameter shape earns its extra parameter and is **not**
redundant with the single-knee logistic — it is the primary closed-form prior.

---

## 7. Models

Ten models are benchmarked under identical splits and metrics:

| Model | Type | Role |
|---|---|---|
| Double-exp fit | empirical curve fit | "physical floor" — if a fixed curve fit on other cells matches the network, the network adds nothing |
| SVR (RBF) | classic ML, non-recurrent | data-driven floor on flattened windows |
| GRU | recurrent baseline | lightweight RNN reference |
| LSTM | recurrent baseline | stronger RNN (more gates/parameters) |
| Transformer (light, 17k) | attention baseline | edge-sized attention model, tuned like every baseline |
| Transformer (heavy, 225k) | attention baseline | high-capacity accuracy reference at published-model scale |
| Neural-ODE | continuous-time PIML | latent-ODE head; "extreme-hardware" comparator |
| Backbone-64 (no physics) | matched-capacity control | the hybrids' identical 41k backbone + time input, physics loss off |
| **Hybrid — route A** | PINN (closed-form prior) | penalizes deviation from the fitted prior curve (the supervisor's preferred "integrated ODE" route) |
| **Hybrid — route B** | PINN (ODE residual) | autograd Verhulst residual dŷ/dt = r·ŷ·(1 − ŷ/K) with **learnable (r, K)** — the thesis centrepiece |

**Hybrid design.** The backbone is a stacked GRU identical in family to the GRU
baseline, and the **matched-capacity control** shares the hybrids' exact 41k
backbone and time input with the physics loss off, so "physics on vs off"
isolates the contribution of the physics — it enters the **loss**, not the
architecture. Every learned model receives the same fixed-scale normalized cycle
coordinate (cycle/500) as an input channel: a BMS knows the elapsed cycle count
online, and the fixed scale carries no information about a trajectory's total
recorded length, so the input is leak-free and symmetric across models. Two
loss-weighting schemes are supported: a **fixed** weight, and **homoscedastic
uncertainty weighting** (learned per-task noise scales).

**Training harness.** Seeded (immediately before model construction, so results
are order-independent), early stopping, learning-rate scheduling on plateau,
gradient clipping, and a tiny-batch overfit sanity check (training loss must
collapse toward zero). Hyperparameters (learning rate, hidden size, sequence
length) were tuned by **Bayesian optimization (Optuna TPE)** on a development
fold; the recorded winning configurations are carried in the notebook as a
development-phase log, and the benchmark hybrids run a larger hidden size (64)
for headroom.

---

## 8. Evaluation protocol

- **Leave-one-cell-out (LOCO)** on NASA: each fold holds out one whole cell for
  test and one for validation; whole cells never mix across splits. **All four
  folds** are evaluated (not a single fold), across **three random seeds**.
- **Cross-dataset transfer:** train on all NASA cells, test on CALCE (a genuine
  domain shift; one test cell available).
- **Metrics:** SOH RMSE and MAE; a **Physical Consistency Score** (fraction of
  consecutive predictions with non-increasing SOH — note the ground-truth
  trajectories themselves score only 0.85–0.90 because regeneration is real, so
  the metric is read as trend plausibility, not accuracy); and PHM prognostic
  metrics (Saxena & Goebel): **α-λ accuracy** and **Cumulative Relative Accuracy
  (CRA)**, evaluated on the pre-EOL region only (post-EOL cycles are masked; the
  Prognostic Horizon degenerates under this masking and is not used as a
  comparison axis). RUL is derived from the predicted SOH trajectory and the EOL
  threshold — a retrospective evaluation of how well the estimated trajectory
  dates end-of-life, applied identically to every model, not an online forecast.
- **Statistics:** bootstrap confidence intervals and a **paired Wilcoxon test at
  the (fold, seed) level (n = 12)**. Seeds within a fold share the test cell and
  only four independent cells exist (capping the smallest attainable cell-level
  p at 0.125), so all test outcomes are read descriptively.

---

## 9. Results and comparative analysis

### Headline benchmark — NASA leave-one-cell-out (mean ± 95 % CI over folds × seeds)

| Model | Params | SOH RMSE | SOH MAE | Phys. cons. | α-λ | CRA | Cross-dataset RMSE |
|---|---|---|---|---|---|---|---|
| Double-exp fit | 4 | 0.0449 ± 0.0065 | 0.0397 | 0.995 | 0.44 | 0.40 | 0.6721 |
| SVR | 6,804 | 0.0492 ± 0.0108 | 0.0371 | 0.699 | 0.44 | 0.49 | 0.1578 |
| Transformer (light) | 17,441 | 0.0379 ± 0.0113 | 0.0346 | 0.789 | 0.49 | 0.09 | 0.1080 |
| Transformer (heavy) | 225,409 | 0.0369 ± 0.0144 | 0.0338 | 0.770 | 0.37 | 0.04 | 0.1277 |
| LSTM | 5,537 | 0.0369 ± 0.0176 | 0.0332 | 0.800 | 0.40 | 0.47 | 0.0925 |
| GRU | 4,161 | 0.0396 ± 0.0075 | 0.0354 | 0.782 | 0.23 | 0.24 | 0.1193 |
| Neural-ODE | 1,249 | 0.0402 ± 0.0152 | 0.0363 | 0.726 | 0.37 | 0.09 | 0.1061 |
| Backbone-64 (no physics) | 41,473 | 0.0377 ± 0.0107 | 0.0345 | 0.879 | 0.26 | −0.06 | 0.0998 |
| **Hybrid (route B)** | 41,473 | **0.0292 ± 0.0100** | **0.0262** | **0.884** | 0.37 | 0.17 | 0.2077 |
| Hybrid (route A) | 41,473 | 0.0382 ± 0.0092 | 0.0347 | 0.876 | 0.32 | 0.06 | **0.0892** |

### Honest reading of the comparison

- **The matched-capacity control settles the attribution.** The control shares
  route B's exact architecture, capacity, and time input, and differs only in the
  physics loss: 0.0377 without physics, 0.0292 with the ODE residual
  (**−22.6 %, 11/12 paired runs, run-level Wilcoxon p = 0.021**, and three of
  four cells with the fourth a near-tie). The gain belongs to the physics term,
  not the capacity.
- **Route B is the accuracy and plausibility leader in-distribution** — lowest
  mean RMSE and the highest physical consistency of the learned models (0.884).
  The purely data-driven baselines cluster at 0.037–0.040; the LSTM fails on the
  atypical B0006 (per-cell RMSE 0.063), exactly where the physics helps most,
  while B0005 is hard for every model.
- **Attention capacity buys no reliable return.** Scaling the Transformer 13×
  (17k → 225k parameters) changes RMSE from 0.0379 to 0.0369 — well inside the
  overlapping intervals — while physical consistency drops and int8 flash grows 8.7×.
- **Cross-dataset transfer is led by route A (0.0892) and the LSTM (0.0925);
  route B fails out of distribution (0.2077).** The mechanism is diagnosed: the
  878-cycle CALCE cell drives the fixed-scale cycle coordinate to 1.76, far
  outside the training range (≤ 0.34), and route B's learned dynamics — which
  depend on that coordinate through the ODE residual — extrapolate poorly.
  Candidate fixes (coordinate saturation, collocation points) are identified.
  Only one CALCE test cell exists, so cross-dataset numbers are indicative only.

### The gap to the heavy Transformer, quantified at the (fold, seed) level (n = 12)

| Comparison | Mean | Reference | Rel. gap | Runs won | Wilcoxon p | Significant |
|---|---|---|---|---|---|---|
| Route B vs Transformer (heavy) — SOH RMSE | 0.0292 | 0.0369 | **−21.0 %** | 5/12 | 0.850 | no |
| Route A vs Transformer (heavy) — SOH RMSE | 0.0382 | 0.0369 | +3.5 % | 4/12 | 0.622 | no |
| Route B vs Transformer (heavy) — Phys. consistency | 0.884 | 0.770 | +14.9 % | **12/12** | **4.9 × 10⁻⁴** | yes* |
| Route B vs Control — SOH RMSE | 0.0292 | 0.0377 | **−22.6 %** | **11/12** | **0.021** | yes* |
| Route B vs Control — Phys. consistency | 0.884 | 0.879 | +0.7 % | 7/12 | 0.168 | no |

\* subject to the independence caution of Section 8 (seeds share test cells; read descriptively).

**Interpretation.** Route B's mean RMSE is 21 % below the heavy Transformer's,
but the mean gap is carried by the Transformer's much larger run-to-run variance
(route B wins only 5/12 individual runs): the advantage is a tighter error
distribution at a much lower mean, not run-by-run dominance, and it is not
statistically significant. What **is** uniform is **physical consistency** —
route B is more plausible on every single run. The defensible claim is
*"more physically plausible on every run, with a sizeable mean accuracy edge the
dataset cannot confirm statistically"* — plus the near-uniform, tested gain over
its own matched-capacity control.

---

## 10. Statistics — the unit of replication matters

Significance is tested at the **(fold, seed) level (n = 12)**, the honest unit of
replication. An earlier analysis paired *per-cycle* residuals (thousands of points)
and reported an absurd p ≈ 5 × 10⁻⁹⁴. That was a **pseudoreplication artifact**:
per-cycle predictions within a cell are strongly autocorrelated, so treating each
as independent massively inflates the effective sample size; and because models
use different sequence lengths, the "paired" residuals were also cycle-misaligned.
Corrected to the run level, the accuracy gap to the Transformer is large in mean
but not significant, while the physical-consistency advantage and the gain over
the matched-capacity control are the uniform signals. Even the run level
overstates independence (seeds within a fold share the test cell; four cells cap
the cell-level p at 0.125), so every test outcome is reported descriptively.

---

## 11. Ablation study

All three ablations run on the **same full protocol as the benchmark** (4 folds
× 3 seeds, benchmark configuration, shared cycle-coordinate channel for every
variant except the explicit "no t" row). The design table is anchored to the
benchmark: its control, route-A and route-B rows re-run the exact benchmark
configurations and reproduce those rows to the fourth decimal. (The original
reduced-protocol ablations — 2 folds × 2 seeds — proved noise-dominated across
reruns and were re-executed at full scale; only the data-scarcity stress test
below remains reduced-protocol and indicative.)

### Design ablation (full protocol)

| Variant | SOH RMSE | Phys. cons. | Train time (s) |
|---|---|---|---|
| Route B · uncertainty weighting | **0.0259** | **0.903** | 3.33 |
| Route B · ODE-residual | 0.0292 | 0.884 | 3.89 |
| Physics OFF (no t) | 0.0312 | 0.879 | 0.97 |
| Physics OFF + t-channel (control) | 0.0377 | 0.879 | 1.13 |
| Route A · uncertainty weighting | 0.0377 | 0.881 | 1.28 |
| Route A · double-exp | 0.0382 | 0.876 | 1.24 |
| Route A · logistic | 0.0383 | 0.876 | 1.10 |

(The metric columns are deterministic across reruns; the train-time column is
wall-clock per fold and varies between runs — the ~3× route-B ratio is the
stable quantity.)

What the decomposition says (paired win counts over the 12 fold×seed runs,
read with the same independence caution as the benchmark):

- **The raw cycle coordinate alone is mildly harmful.** The backbone scores
  0.0312 without it and 0.0377 with it (better with it on only 4/12 runs,
  p = 0.27); it pays off only once the ODE residual constrains how the network
  may use it. The clean physics attribution remains the matched-input
  comparison (0.0377 → 0.0292, 11/12 runs, p = 0.021); against the stronger
  no-t backbone, route B keeps a mean advantage (−6.6 %) without run-level
  dominance (7/12, p = 0.47).
- **Loss weighting (refined negative).** Uncertainty weighting has no accuracy
  effect on route A (0.0377 vs 0.0382, 5/12 runs, p = 0.85). On route B it
  posts the best mean in the table (0.0259, −11.1 %) — but the mean gap is
  carried by run-to-run variance (6/12 runs, p = 0.62), so it is not a
  dependable accuracy gain. Its one fairly uniform effect is physical
  consistency on route B (0.903 vs 0.884, 10/12 runs, p = 0.019). The fixed
  weight stays in the headline models; the uncertainty-weighted route B is the
  natural configuration to validate on a larger fleet.
- **The two closed-form priors are interchangeable inside route A** (0.0382 vs
  0.0383): as a soft penalty, the prior's exact shape matters far less than
  whether its parameters can adapt (route B).
- **Route B costs ~3× the training time** of the no-physics baseline — paid
  offline; the deployed inference network is unchanged.

### Other ablations

- **Feature richness (aggregation).** On the full protocol the minimal
  two-feature set — precisely the two near-proxy features (discharge time,
  mean voltage) — scores a *lower* mean RMSE than the full 8-feature set
  (0.0315 vs 0.0382): direct evidence for the feature–label-proximity caution
  of Section 5. The richer set earns its place through the ICA features'
  physical interpretability, not pointwise RMSE. (This is a feature-count
  surrogate for the per-second→per-cycle information trade-off, not a literal
  within-cycle study.)
- **Data volume.** Subsampling 25–100 % of each training cell's windows shows
  **no clean trend** (0.034–0.039 band): on this small, clean dataset the model
  is **not data-starved**. Overlapping windows retain nearly full life coverage
  at low fractions, so this understates the difficulty of a genuinely shorter
  record.
- **Data scarcity vs the Transformers (10–100 %).** Still on the reduced
  protocol and noise-dominated; the dependable capacity conclusion is the
  full-benchmark one (Section 9): 13× more attention parameters buy no
  reliable accuracy return.

---

## 12. Edge deployment & Cortex-M4 footprint

Models are quantized to int8 (PyTorch dynamic quantization, qnnpack) and profiled
against the M4 budget. Sizes and RAM are **measured**; the M4 latency is an
**analytic** estimate (MAC count ÷ clock at one MAC/cycle); activation RAM is the
single largest intermediate tensor (a lower bound on the true working set). The
MAC counter instruments linear and recurrent layers only, so the Transformers'
attention matrix products are *not* counted — a bias in the Transformers' favour
that makes the hybrid's latency advantage conservative.

| Model | Params | int8 flash | Activation RAM | MACs | Est. M4 latency | Fits M4 |
|---|---|---|---|---|---|---|
| Deployable hybrid (route B, hidden 32) | 11,009 | 20.2 KB | 1.25 KB | 101,328 | 1.27 ms | yes |
| GRU (small) | 4,161 | 9.2 KB | 1.25 KB | 39,392 | 0.49 ms | yes |
| Transformer (light) | 17,441 | 70.0 KB | 2.5 KB | 84,832 | 1.06 ms | yes |
| Transformer (heavy) | 225,409 | 611.1 KB | 7.5 KB | 1,114,656 | 13.93 ms | int8 only |

**Matched-size accuracy — the trade-off stated plainly.** Re-evaluated on the
same full four-fold leave-one-cell-out protocol, the deployable 11k model scores
**SOH RMSE 0.0469 ± 0.0217 with physical consistency 0.877**. Shrinking from 41k
to 11k parameters therefore **costs accuracy** (0.0292 → 0.0469) and adds
substantial fold-to-fold variance. Against the Transformers the deployed model is
statistically indistinguishable on accuracy (intervals overlap broadly) while
keeping the highest physical consistency of the deployment candidates, at 3.5×
less flash than the light Transformer and 30× less than the heavy one (with an
11× latency margin over the latter).

> On the three hybrid parameter counts that appear in the study — all intentional:
> development/HPO used the smaller hidden size (≈11 k); the head-to-head benchmark
> deliberately uses a larger hidden size (≈41 k) for headroom; the deployed model
> returns to ≈11 k and its accuracy cost is measured and reported above.

**Honest framing of "lightweight."** The deployed model's case is **physical
consistency + footprint** — plausibility per kilobyte — not peak accuracy.
Applications that can afford the benchmark-size model's memory (≈166 KB fp32)
should deploy that model instead.

---

## 13. Key findings (honest summary)

1. **The data is verified.** Cycle counts (168/168/168/132), initial capacities,
   B0006 starting above rated (SOH 101.8 %), and the CALCE 878-cycle trajectory
   (1.138 → 0.302 Ah, 260,931 records, duplicate dropped, outliers filtered) all
   reproduce exactly.
2. **Real physics parameters** replace earlier placeholders; the double-exponential
   beats the logistic by information criterion on every cell (read as evidence for
   the more flexible shape, not as a literal SEI/LAM decomposition).
3. **The learnable ODE-residual hybrid (route B) leads the benchmark** — the
   lowest mean SOH RMSE (0.0292 ± 0.0100) and the highest physical consistency of
   the learned models (0.884), more physically consistent than the heavy
   Transformer on 12/12 runs. The **fixed prior (route A) can hurt atypical
   cells** — and, conversely, transfers best across datasets.
4. **The physics genuinely helps, and the control proves it.** The
   matched-capacity control (identical 41k backbone + time input, physics off)
   scores 0.0377 vs route B's 0.0292: −22.6 %, on 11/12 paired runs (run-level
   p = 0.021). The gain is attributable to the physics residual, not capacity or
   the time input.
5. **Statistics are tested at the right level.** Route B's mean accuracy edge over
   the heavy Transformer (−21.0 %) is *not* significant (5/12 runs; the mean gap
   reflects the Transformer's variance), while its physical-consistency edge holds
   on every run — claim "significantly more plausible," not "significantly more
   accurate."
6. **Negative results are reported with mechanisms:** uncertainty weighting
   brings no reliable *accuracy* benefit on either route (on route B its better
   mean, 0.0259, is carried by run-to-run variance — 6/12 runs — while its one
   uniform effect is a physical-consistency gain, 0.903 on 10/12 runs); the raw
   cycle coordinate alone makes the unconstrained backbone worse
   (0.0312 → 0.0377) and pays off only under the ODE residual; route B fails
   out of distribution (0.2077) because its cycle coordinate extrapolates
   beyond the training range; and shrinking to the 11k deployment size costs
   accuracy (0.0469 ± 0.0217) — the deployed model's case is plausibility per
   kilobyte, within 2 % of the M4 flash budget.

---

## 14. Corrections to the earlier project report

- Chemistry: CALCE is **LCO**, not LFP/NMC; NASA→CALCE is **cross-condition**, not
  cross-chemistry.
- CALCE initial capacity ≈ **1.14 Ah** (the earlier ≈0.88 Ah figure is mid-life).
- Record count corrected to **260,931** (an earlier count double-counted the
  duplicate file).
- All result tables, confidence intervals, and significance tests are **real and
  measured**, replacing placeholder values; the significance test is computed at
  the run level rather than per cycle.
- The NASA charge↔discharge pairing was corrected from positional to
  sequence-order pairing (the ICA features were 1–2 cycles stale on most cycles).
- The cycle-coordinate input was corrected from a per-trajectory normalization
  (which leaked each test cell's recorded life) to a fixed scale supplied
  symmetrically to every model.
- The headline is reframed from a blanket "the hybrid always wins" to the
  evidence-backed claim about the physics-attributable accuracy gain at matched
  capacity, physical plausibility, and footprint.

---

## 15. Limitations and honest caveats

- **Few cells.** Four NASA cells and one CALCE cell limit statistical power;
  in-distribution accuracy differences between the best models are not significant,
  and cross-dataset rankings (one cell) are indicative only.
- **Near-proxy features.** Under constant-current lab cycling, discharge time and
  ICA area approximate the same-cycle capacity that defines the label; a
  production BMS rarely observes complete constant-current discharges. An ablation
  with same-cycle capacity proxies excluded (lagged features) is future work.
- **Cycle-coordinate extrapolation.** The fixed-scale coordinate leaves the
  training range on trajectories much longer than the training cells — the
  diagnosed mechanism behind route B's cross-dataset failure. Candidate fixes:
  saturate the coordinate at the training maximum, or evaluate the ODE residual on
  collocation points across (and beyond) the training range, as in standard PINN
  practice. The profiling path zeroes the coordinate; the exported ONNX takes it
  as an explicit input matching the evaluated model.
- **Retrospective RUL.** The PHM metrics score RUL curves derived from each
  model's full predicted SOH trajectory — how well the estimated trajectory dates
  end-of-life in hindsight, identically for every model — not an online forecast.
- **M4 latency is analytic**, not measured on a physical board; on-board timing is
  future work. Activation-RAM is a single-tensor lower bound; attention matmuls
  are excluded from MAC counts (conservative toward the hybrid).
- **Aggregation ablation** is a feature-count surrogate, not a literal
  within-cycle-vs-per-cycle representation study.
- **Hyperparameter-selection overlap.** Configurations were tuned once on a fixed
  development fold whose cells later serve as held-out test cells in LOCO; with
  four cells a fully nested protocol is impractical, so this is disclosed rather
  than hidden.

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
4. A full run trains many small models on CPU; the two most recent full runs
   measured 20–22 minutes on an 8-core Apple-silicon laptop (the heavy
   225k-parameter Transformer and the full-protocol ablations dominate the
   wall time; budget up to ~an hour on older hardware). A much faster
   single-seed quick mode is available via one flag in the setup cell; the
   canonical multi-seed numbers are printed alongside as a reference regardless
   of mode.

The notebook is **self-contained**: it depends only on the raw battery dataset and
defines every function it uses internally.

**Required data layout.** The notebook searches upward from its own folder for a
`data/` directory containing both `data/NASA/` and `data/CALCE/`. Note that the
NASA ICA features are parsed from the **raw `.mat` files** — the per-cycle CSVs
alone are not sufficient:

```
data/
├── NASA/
│   ├── B0005_cycle_summary.csv          (and B0006/B0007/B0018)
│   ├── B0005_discharge_timeseries.csv   (and B0006/B0007/B0018)
│   └── raw data/
│       └── B0005.mat  B0006.mat  B0007.mat  B0018.mat   ← required for ICA
└── CALCE/
    └── CS2_35 channel CSV exports
```

Without the `.mat` files the NASA feature-building cell raises
`FileNotFoundError`.
