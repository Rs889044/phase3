# Code Demo — Voice-Over Script (10 minutes)

**Setup before recording (do once):**
1. Run `Complete_Project_Notebook.ipynb` fully the day before (keep all outputs — do **not** clear them).
2. Open both notebooks in tabs: `Demo_Overview.ipynb` first, `Complete_Project_Notebook.ipynb` second.
3. Zoom the notebook to 125–150% so code is readable.
4. Before hitting record, run the main notebook from the top up to "Windowing ready" (~4 min), so the live cells work on camera.

Speak slowly and calmly. The spoken text below is ~1,200 words — about 10 minutes at a relaxed pace. `[SHOW: …]` lines are screen directions, not spoken.

---

## PART A — Overview notebook (`Demo_Overview.ipynb`) — 0:00 to 3:15

### 1. Project overview — 0:00–0:50

`[SHOW: title cell, then Section 1]`

> Hello. I am Rajat Sharma, roll number CH24M559, M.Tech in Industrial AI at IIT Madras. This demo covers my thesis: estimating Remaining Useful Life and State of Health of EV batteries with a hybrid physics-informed deep learning framework.
>
> The problem: batteries degrade, and the Battery Management System must estimate health and remaining life accurately. Data-driven models are accurate but physically unconstrained and too large for BMS chips; physics models are interpretable but too slow. My solution is a small GRU network whose loss is regularized by a battery degradation ODE. The target is on-board deployment: an ARM Cortex-M4 chip with one megabyte of flash.
>
> Headline result: the hybrid gets the lowest error among ten models, a matched no-physics control proves the gain comes from the physics loss, and the deployed model needs only twenty kilobytes — two percent of the budget.

### 2. Languages and frameworks — 0:50–1:20

`[SHOW: Section 2 table]`

> Everything is Python 3.12.8. The core framework is PyTorch 2.5.1 — models, autograd for the physics residual, and int8 quantization. NumPy 2.2.1 and pandas 2.2.3 for numerics, SciPy 1.15.1 for curve fitting and statistics, scikit-learn for scaling and the SVR baseline, Matplotlib for figures, and Optuna 4.9 for Bayesian hyperparameter search. No TensorFlow, no LangChain, no Hugging Face — this is not an LLM project.

### 3. Compute resources — 1:20–1:45

`[SHOW: Section 3]`

> Everything runs locally on my laptop: Apple M2, eight cores, eight gigabytes of RAM, no GPU. That is a design point — the thesis is about models small enough for embedded hardware, so a CPU is enough. The dataset is 176 megabytes, and a complete end-to-end run takes twenty to twenty-two minutes, measured.

### 4. Dataset details — 1:45–2:25

`[SHOW: Section 4 tables]`

> Two public run-to-failure datasets, both lithium cobalt oxide. NASA: four 18650 cells, 636 labeled discharge cycles, parsed from raw MATLAB files. CALCE: one prismatic cell, 878 cycles, assembled from 261 thousand raw records. This is regression — no classes, no tokens. The label is per-cycle State of Health; RUL comes from the eighty-percent threshold with a five-cycle persistence rule.
>
> Preprocessing includes timestamp ordering, an MD5-confirmed duplicate removal, outlier filtering, and eight engineered features per cycle. The split is leave-one-cell-out: whole cells never mix between train, validation, and test, and the scaler is fit on training cells only — no leakage. Each fold runs with three seeds.

### 5. Tools and environment — 2:25–2:40

`[SHOW: Section 5 table]`

> In short: PyTorch on CPU, fully local, no cloud, no APIs. Quantization is PyTorch dynamic int8 — sizes measured, not estimated — and the deployed model is exported to ONNX.

### 6. Code organization — 2:40–3:00

`[SHOW: Section 6 folder tree]`

> The implementation is one self-contained notebook, 59 cells — every function defined inside, so anyone can reproduce every number from one file plus the raw data. A single CONFIG dictionary acts as the configuration file, and every figure and table auto-saves to disk — the thesis tables come straight from these outputs.

### 7. Pipeline design — 3:00–3:15

`[SHOW: Section 7 data-flow diagram]`

> The pipeline is fully automatic, top to bottom, in five phases: data foundation, features and physics, models, evaluation, deployment. Training is offline; only the small quantized network gets deployed. Now, the code.

---

## PART B — Main notebook (`Complete_Project_Notebook.ipynb`) — 3:15 to 10:00

### Setup and reproducibility — 3:15–4:00

`[SHOW: setup cell + printed output; point at set_seed and CONFIG]`

> Here is the setup cell — the printed versions, three seeds, 120 epochs, and the CONFIG dictionary with the four folds and locked hyperparameters. One key practice: every model is seeded immediately before construction, so results do not depend on execution order.

`[SHOW: scroll to the sanity-check cell and RUN IT LIVE — instant]`

> Let me run the data-invariant checks live. These assertions lock the cleaned data to verified ground truth — cycle counts, the 260,931 CALCE records, the duplicate confirmed by hash. All pass.

### Data foundation — 4:00–4:40

`[SHOW: find_eol briefly, then the CALCE assembly cell, then the capacity-fade figures]`

> Labels are defined once: end-of-life requires SOH below eighty percent for five consecutive cycles, so noise cannot trigger it. The CALCE assembly handles the hard parts — internal-timestamp ordering, cycle-index offsets, the duplicate file, and capacity computed as the increment of the cumulative column, which removes fifty-amp-hour outliers.
>
> The result: four NASA cells with the regeneration sawtooth — exactly what the physics must tolerate — and CALCE's full 878-cycle fade.

### Features and physics priors — 4:40–5:20

`[SHOW: dQ/dV aging figure, then the prior-fit and AIC outputs]`

> For features, the charge curve is smoothed with Savitzky–Golay, differentiated to dQ/dV, and the dominant peak extracted. This figure shows the peak shrinking and shifting with age — a known electrochemical health signature.
>
> Two closed-form degradation laws are then fitted per cell — logistic and double-exponential, R-squared up to 0.99 — and an AIC analysis shows the double-exponential earns its extra parameter on all five cells.

### Models and training — 5:20–6:40

`[SHOW: models cell, then VerhulstODEResidual class]`

> Ten models: curve fit and SVR as floors, GRU, LSTM, two Transformer scales, a Neural-ODE, a matched-capacity control, and the hybrid in two variants. The physics enters the **loss**, not the architecture.
>
> This class is the centre of the thesis: the Verhulst ODE residual. Autograd computes the derivative of predicted SOH with respect to the cycle coordinate, and the loss penalizes deviation from r y times one minus y over K — with r and K **learnable**. That is route B; route A uses the fixed fitted curve instead.
>
> Training: Adam, learning rate 0.002, batch 64, up to 120 epochs, early stopping at patience 15, gradient clipping, physics weight 0.1 — tuned with Optuna. The benchmark hybrid has 41,473 parameters; the deployed one, 11,009.

`[SHOW: RUN the three-variant live-training cell (~15–30 s)]`

> Training live now — the same backbone with physics off, route A, and route B, on one fold. It runs in seconds on CPU. Done — and route B prints its learned physics: r about minus 0.9, K about 0.87. A degradation law learned from data.

### Evaluation and results — 6:40–7:50

`[SHOW: T1 benchmark table]`

> The full evaluation is four folds times three seeds for ten models — twenty minutes — so I show the saved output. Route B wins: RMSE 0.0292, and the highest physical consistency of any learned model. The critical row is the matched control — identical backbone, physics off: 0.0377. Same capacity, same inputs, so the twenty-three percent gain belongs to the physics.

`[SHOW: T2 paired-statistics table]`

> Statistics are honest, at the run level with paired Wilcoxon tests. Against the heavy Transformer the mean gap is twenty-one percent but **not** significant — only five of twelve runs. The control comparison holds on eleven of twelve, p equal 0.021. And one negative result is reported openly: route B degrades on zero-shot transfer to CALCE, because its cycle coordinate extrapolates beyond the training range — the mechanism is diagnosed in the thesis.

### Ablations — 7:50–8:20

`[SHOW: design-ablation table + tie-in printout]`

> The ablations run on the same full protocol, and as a built-in check their control and route rows re-run the benchmark configurations exactly — the tie-in printout shows they match. Finding: uncertainty weighting gives no reliable accuracy gain on either route, so the fixed weight is kept.

### Deployment — 8:20–9:20

`[SHOW: footprint table, then ONNX printout]`

> Deployment mode: local execution with an embedded target — no API, no cloud. The deployed hybrid is quantized to int8 and **measured**: 20.2 kilobytes of flash — two percent of the Cortex-M4 budget — 1.25 kilobytes of RAM, and about 101 thousand multiply-accumulates per window, an analytic 1.27 milliseconds at 80 megahertz. Thirty times less flash than the heavy Transformer. The ONNX export is 49 kilobytes.
>
> Performance summary: one to four seconds training per fold for small models, twenty-two minutes for the whole pipeline, and the accuracy cost of the small deployed size is stated openly in the thesis.

### LLM checklist and closing — 9:20–10:00

`[SHOW: final takeaways cell]`

> On the LLM checklist: this project uses no language model, no prompts, no RAG, no embeddings, no agents — so token usage, guardrails, and generation evaluation are not applicable. Evaluation here is classical: cross-validation, paired statistics, prognostic metrics.
>
> To close: a reproducible, leak-free pipeline from 261 thousand raw records to a physics-informed model that beats its matched control by twenty-three percent and fits in twenty kilobytes — with negative results reported alongside the positive ones. Thank you.

---

## Suggestions

1. **Timing.** ~1,200 words ≈ 10 minutes at a relaxed pace. If you still run long, cut the AIC sentence and the negative-result sentence first. Never cut the control-row explanation or the deployment numbers — they are your strongest points.
2. **Live runs.** Two safe moments: the assert cell (instant) and the three-variant training demo (~15–30 s). Never run the full evaluation cell on camera.
3. **Kernel state.** The live cells need earlier cells executed — run up to "Windowing ready" before recording.
4. **"Personally explain."** A 10-second webcam intro ("I am Rajat Sharma…") makes the requirement indisputable; voice-over covers the rest.
5. **One dry run.** Read it aloud once with a timer; rephrase any sentence you stumble on into your own words.
