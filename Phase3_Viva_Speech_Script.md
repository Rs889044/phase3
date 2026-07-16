# Viva Speech Script — Short & Concise (~10–11 min at a slow pace)

**How to use:** This is a tighter version — shorter sentences, every key number kept, nothing essential lost. It runs about 10–11 minutes even at a slow, calm pace (~115 words/min), leaving buffer inside your 15-minute limit. Pause briefly after each big number.

---

## Slide 1 — Title  (≈ 30 s)

Good morning. I am Rajat Sharma, roll number CH24M559, from M.Tech in Industrial Artificial Intelligence.

My thesis estimates the Remaining Useful Life and State of Health of EV batteries using a hybrid physics-informed deep learning model — a small, accurate model that respects battery physics and is light enough to run on a real battery-management chip.

---

## Slide 2 — Introduction  (≈ 45 s)

A battery management system needs two numbers: State of Health — capacity now versus rated — and Remaining Useful Life — cycles left until 80 percent health.

Getting them wrong is costly. Over-estimate, and we charge a weak cell too hard — risking plating and fire. Under-estimate, and we scrap healthy packs early.

As the figure shows, ageing is non-linear: a slow fade, then a sharp knee. My key constraint: train offline, but run on a small on-board microcontroller.

---

## Slide 3 — Literature Survey  (≈ 50 s)

Prior work falls into three groups.

Physics models are interpretable but heavy — full PDE models need over twenty parameters. Data-driven networks are accurate with enough data, but can predict impossible "self-healing" and are memory-hungry. Hybrid methods combine both, yet most embed heavy PDEs.

Three gaps remained, and my thesis fills them: compare fixed versus learnable physics parameters fairly; test uncertainty weighting honestly against a fixed weight; and give a measured — not estimated — deployment profile on real hardware.

---

## Slide 4 — Problem Statement  (≈ 40 s)

The definitions are on screen. State of Health is capacity now over rated capacity. The task is to predict it from a window of recent cycle features. End of life is when health drops to 80 percent and stays there for five cycles — that rule blocks false alarms. Remaining Useful Life is simply the cycles left to that point.

So the question is: high accuracy and physical consistency, on tiny hardware, from limited noisy data — all at once.

---

## Slide 5 — Physics as a Soft Constraint  (≈ 50 s)

I add physics into the loss as a soft rule, using two simple laws. The Verhulst law gives a smooth, self-limiting fade. The double-exponential law adds two timescales, which captures the knee.

The total loss has two parts: a normal data loss, plus a physics loss. Using automatic differentiation, I take the model's own slope and pull it toward what the law expects. Being soft, it discourages self-healing but still allows genuine regeneration.

---

## Slide 6 — The Framework  (≈ 45 s)

Here is the pipeline. Raw voltage, current and temperature become physics-based features — with smoothing applied before differentiation. These feed a GRU.

Why a GRU? Two gates instead of the LSTM's three, and linear memory instead of a Transformer's quadratic attention — so it stays small. Hyperparameters were tuned with Optuna. Sizes run from four thousand parameters up to forty-one thousand, and eleven thousand for deployment.

---

## Slide 7 — Two Physics Routes  (≈ 50 s)

The physics can enter two ways, and I let the data decide.

Route A fixes the law: fit it once on training cells, then gently pull predictions toward that curve. Cheap, but it assumes new cells behave like old ones.

Route B makes the law learnable: its parameters train with the network, so the physics adapts. It costs more training time, paid offline.

I also tested the loss balance two ways — a tuned fixed weight versus learnable uncertainty weighting.

---

## Slide 8 — Datasets  (≈ 45 s)

I use two public datasets. NASA has four cells famous for capacity regeneration — cell B0005 shows 36 upward "healing" jumps, a perfect test of physical consistency. CALCE is one long cell with a clear knee, used as a zero-shot generalization test.

Both share the same eight per-cycle features plus a cycle counter. Two honesty points: labels are kept raw, so models face real regeneration, and all scaling uses training cells only — no leakage.

---

## Slide 9 — Experimental Setup  (≈ 45 s)

The protocol is strict leave-one-cell-out: four folds, three seeds, twelve runs. Whole cells never mix, and every model gets the same inputs — an earlier leak was found and fixed.

Ten models compete under identical conditions, including a key control: the exact hybrid network with physics switched off. Metrics cover accuracy, a physical-consistency score, standard prognosis metrics, paired significance tests, and a hardware profile.

---

## Slide 10 — Results, Part 1  (≈ 55 s)

The learnable hybrid, Route B, gives the lowest error in the benchmark — 0.0292 — and the best physical consistency of any learned model.

The key number is the comparison with the matched control — same network, physics off — which scores 0.0377. So adding physics to the very same network cuts error by 22.6 percent, on eleven of twelve runs, p equals 0.021. That proves the gain comes from physics, not size.

Route B also beats the fixed prior on every cell, and its learned parameters stay readable.

---

## Slide 11 — Results, Part 2  (≈ 60 s)

Now fair comparisons and honest negatives.

Against a Transformer thirteen times larger, my error is 21 percent lower on average — but not statistically significant with only four cells. What is uniform: better physical consistency on all twelve runs.

Two honest negatives: uncertainty weighting gave no reliable accuracy gain, and Route B fails the cross-dataset transfer because the cycle counter goes out of range — a known, fixable issue.

Finally, deployment — measured, not estimated: just 20.2 kilobytes and about two percent of a Cortex-M4, thirty times smaller than the big Transformer.

---

## Slide 12 — Conclusions  (≈ 45 s)

In short: learnable physics improves a small network where extra size does not — 22.6 percent lower error at the same size, with the most plausible predictions. What matters is that the physics can adapt.

And it deploys for real — about two percent of a small chip. The limits are stated openly, not hidden.

Future work: fix the cross-dataset case, test on real hardware, and extend to other chemistries.

Physics makes a small network smart; smallness makes it deployable. Thank you — I welcome your questions.

---

## Timing summary (slow pace ≈ 115 words/min)

| Slide | Time | Running total |
|---|---|---|
| 1. Title | 0:30 | 0:30 |
| 2. Introduction | 0:45 | 1:15 |
| 3. Literature | 0:50 | 2:05 |
| 4. Problem | 0:40 | 2:45 |
| 5. Physics | 0:50 | 3:35 |
| 6. Framework | 0:45 | 4:20 |
| 7. Two routes | 0:50 | 5:10 |
| 8. Datasets | 0:45 | 5:55 |
| 9. Setup | 0:45 | 6:40 |
| 10. Results 1 | 0:55 | 7:35 |
| 11. Results 2 | 1:00 | 8:35 |
| 12. Conclusions | 0:45 | **≈ 9:20** |

At a slow, deliberate pace with pauses this comfortably lands near 10–11 minutes — safely inside 15.

**Delivery tips:** pause after 0.0292, −22.6%, and 20.2 KB. On Slide 10, point to the green bar as you say "lowest error." Speak the honest-negatives slide calmly — that candour is what earns examiner trust.
