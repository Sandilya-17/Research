# v2 Revision — Findings Report

This report documents what changed in the code (`v2/`) relative to the
original submission (`v1_original/`, kept for reference), what the four
requested fixes actually found, and what that means for the paper.

**Headline: once the four fixes are applied, the hybrid ML+VQE pipeline no
longer beats — or even matches — plain ZNE, and in the fidelity/KL metrics
it is also worse than doing nothing (the raw noisy output).** This is an
important, if unwelcome, finding, and section 5 below discusses what to do
about it.

---

## 0. Two bugs found and fixed while implementing the requested changes

Before getting to the four numbered items, two implementation bugs surfaced
that made the v1 numbers unreliable independent of anything already flagged
in review:

**(a) Label degeneracy (the reviewer's core concern), confirmed empirically.**
In `v1_original/.../main.py`, every training sample's label `y` was
`ideal_vec` for the *same* fixed circuit and target (`"1011"`), and the
simulator was always called with `seed_simulator=SEED` (a hardcoded
constant). Both the circuit and the seed being fixed means the "ideal"
execution is bit-for-bit deterministic — so literally every one of the 400
training labels was an identical vector. We verified this directly:
`len(np.unique(y, axis=0))` on a reconstruction of the v1 training loop
returns **1**. A model can hit near-zero MSE against a constant target by
just memorizing that constant, without learning any real noise-correction
map. This is fixed in v2 (see §1).

**(b) The adaptive ensemble weight leaked the true label.** v1's Eq. (9-10)
sets the RF/MLP blend weight to `F(p_ideal, p_MLP)` — the fidelity between
the MLP's prediction and the *true, unknown-at-inference-time* ideal
distribution. That is the exact quantity the whole pipeline is trying to
recover; using it to *tune the pipeline's own output* is not something you
can do at real inference time, and it inflates the reported numbers. v2
replaces this with a leak-free rule (`ml_models.ensemble_predict_no_leak`)
that weights by how much the RF and MLP predictions already agree with each
other, which requires no knowledge of the ground truth.

**(c) The VQE "hardware fidelity" feature was noise-independent.** v1's
`run_vqe()` used a noiseless `Estimator`, so the feature took the same
value regardless of the noise level being corrected for — it could not
possibly have carried the noise information the paper attributes to it.
v2's `vqe.py` runs the VQE probe under the same noise model as the Grover
circuit at each noise level, so the feature now genuinely varies with λ
(confirmed: `run_vqe(0.0)` and `run_vqe(0.1)` differ).

All code is commented at the point of each fix with a short explanation of
what was wrong and why.

---

## 1. Fix: resolve label degeneracy with multiple marked states

`config.py` now defines 10 training targets and 3 held-out test targets
(4-qubit case). `build_dataset()` builds one Grover circuit per target, so
`y` varies genuinely across the training set:

```
Training set shape: X=(900, 18), y=(900, 16)
Unique label rows in y: 10   (v1: 1)
```

Models are evaluated only on the 3 **held-out** targets, which were never
seen in any training sample — a genuine generalization test, not just a
different noise realization of a memorized circuit.

## 2. Fix: repeat evaluation over multiple seeds, report mean ± std

For every held-out target, 20 independent shot-noise seeds are drawn at the
test noise level λ=0.10 (`main_v2.py`, `N_EVAL_SEEDS=20`). All headline
numbers below are **mean ± std over 3 targets × 20 seeds = 60 trials**,
not a single-seed point estimate. Per-target and pooled breakdowns are both
saved (`summary_per_target_mean_std.csv`, `summary_pooled_mean_std.csv`).

## 3. Fix: ablation (VQE on/off; RF-only vs MLP-only vs ensemble)

Four model configurations are trained on the identical data and evaluated
on the identical held-out set:

| Variant | Description |
|---|---|
| `RF_only_with_VQE` | Random Forest alone |
| `MLP_only_with_VQE` | MLP alone |
| `ensemble_no_VQE` | leak-free adaptive ensemble, VQE feature dropped |
| `ensemble_with_VQE` | leak-free adaptive ensemble, full 18-dim feature (v1's config) |

## 4. Fix: scaling check (5 qubits / 32 outcomes)

`scaling_v2.py` repeats the identical protocol at 5 qubits with a smaller,
proportionate training corpus (6 train targets, 4 noise levels, 10 reps =
240 samples for a 32-dim label, vs. 900 samples for a 16-dim label at 4
qubits — 7.5 vs. 56.25 samples per label dimension), to make the O(2ⁿ)
scaling cost concrete rather than asserted.

---

## Results

### 4-qubit generalization test (mean ± std, 3 held-out targets × 20 seeds, λ=0.10)

| Method | Fidelity ↑ | KL ↓ | MSE ↓ |
|---|---|---|---|
| Noisy (no mitigation) | 0.758 ± 0.026 | 0.665 ± 0.098 | 0.0114 ± 0.0010 |
| **ZNE** | **0.879 ± 0.017** | **0.272 ± 0.042** | **0.0064 ± 0.0008** |
| RF-only (+VQE) | 0.662 ± 0.023 | 1.005 ± 0.045 | 0.0146 ± 0.0017 |
| MLP-only (+VQE) | 0.680 ± 0.011 | 0.987 ± 0.048 | 0.0135 ± 0.0004 |
| Ensemble, no VQE feature | 0.690 ± 0.005 | 0.944 ± 0.026 | 0.0133 ± 0.0002 |
| Ensemble, +VQE feature (v1's config) | 0.680 ± 0.011 | 0.986 ± 0.047 | 0.0135 ± 0.0004 |

(See `results/main_4qubit_run/pooled_comparison.png` for a bar-chart view,
and `summary_per_target_mean_std.csv` for the target-by-target breakdown,
which shows the same ranking holds for every individual held-out target.)

### 5-qubit generalization test (mean ± std, 2 held-out targets × 10 seeds, λ=0.10)

| Method | Fidelity ↑ | KL ↓ | MSE ↓ |
|---|---|---|---|
| Noisy | 0.852 ± 0.008 | 0.421 ± 0.030 | 0.0019 ± 0.0001 |
| **ZNE** | **0.858 ± 0.015** | **0.391 ± 0.062** | **0.0018 ± 0.0001** |
| Hybrid ensemble (+VQE) | 0.822 ± 0.009 | 0.469 ± 0.037 | 0.0020 ± 0.0001 |

---

## 5. What this means for the paper

1. **The v1 headline numbers (fidelity 0.74→0.99, marked-state prob.
   0.05→0.48) do not survive a genuine held-out test.** They were most
   likely produced predominantly by the label-degeneracy bug in §0(a),
   compounded by the label-leaking ensemble weight in §0(b). Once both are
   fixed, plain ZNE outperforms the learned ensemble at both 4 and 5
   qubits, on every metric, with the gap outside one standard deviation.

2. **The ablation is informative even though the headline result is
   negative.** Ensemble-without-VQE slightly *beats* ensemble-with-VQE on
   every metric at 4 qubits, and RF-only is the worst single model. This
   suggests: (i) the current 18-dim feature representation and ~900-sample
   training corpus are not enough for the MLP/RF to learn a transferable
   noisy→ideal correction across different marked states, and (ii) the VQE
   feature, once made properly noise-dependent, adds noise to the
   prediction rather than signal, at least with this small a training set
   and this noisy an optimizer (COBYLA, 50 iterations, stochastic random
   restarts).

3. **This changes the paper's contribution.** As written, the paper claims
   a positive result ("hybrid ML+VQE recovers near-ideal search output").
   The corrected experiments don't support that claim in a genuine
   generalization setting. Two honest paths forward:
   - **Reframe as a cautionary/methodological paper**: "we identify and
     empirically confirm a label-degeneracy failure mode common to
     ML-based quantum error mitigation studies with single-circuit
     training data, show it inflates reported gains, and provide a
     corrected multi-target evaluation protocol other groups can reuse."
     This is a legitimate, publishable contribution on its own, and this
     revision essentially is the first draft of that paper.
   - **Do more work before re-claiming superiority over ZNE**: substantially
     larger and more diverse training corpora (more targets, more circuit
     families), feature engineering beyond the raw 16-dim histogram,
     and/or a fundamentally different model class (e.g., a
     permutation/structure-aware model that exploits the fact that the
     ideal distribution is a simple deterministic function of the target
     bitstring) would be needed before the "hybrid ML beats ZNE" claim
     could be re-tested honestly.
   Either way, the original single-target, single-seed, label-leaking
   experimental design should not be published as-is, and per your point 4,
   this is exactly the kind of scope that's a much more natural fit for a
   workshop/short paper than a full IEEE Transactions submission — the
   value now is in the methodological finding, not in a large positive
   quantitative result.

4. All raw data, seeds, and configs needed to reproduce every number above
   are saved under `results/main_4qubit_run/` and
   `results/scaling_5qubit_run/`, including per-trial CSVs
   (`raw_eval_records.csv`) so you (or a reviewer) can re-aggregate or
   re-slice the statistics independently.

---

## Note on reading the CSVs

Target bitstrings like `"0100"` have a leading zero. If you load
`raw_eval_records.csv` with plain `pandas.read_csv(...)`, pandas will
auto-infer that column as an integer and silently drop the leading zero
(you'll see `100` instead of `0100`). This is purely a CSV-reading quirk,
not a bug in the simulation itself (the correct 4-character bitstring was
used throughout every run). Load with
`pd.read_csv(path, dtype={"target": str})` to preserve it.

## File map


```
v2/
  config.py          - all experiment parameters (targets, noise grids, seeds)
  grover.py           - n-qubit, n-target Grover circuit builder
  noise.py            - unchanged noise model
  sim.py              - simulator execution + ZNE, now with explicit seeds
  vqe.py              - noise-aware VQE probe (rewritten on EstimatorV2)
  dataset.py           - counts <-> probability vector helpers
  metrics.py           - fidelity / KL / MSE (unchanged)
  ml_models.py         - RF/MLP training + leak-free adaptive ensemble
  pipeline.py          - shared dataset-build + evaluation logic
  main_v2.py           - 4-qubit generalization + stats + ablation experiment
  scaling_v2.py        - 5-qubit scaling experiment
  results/
    main_4qubit_run/
      raw_eval_records.csv            - every individual trial (target, seed, variant, metrics)
      summary_per_target_mean_std.csv - mean+-std per (target, method)
      summary_pooled_mean_std.csv     - mean+-std pooled across targets
      pooled_comparison.png           - bar chart of the table above
      config_used.json                - exact parameters used for this run
    scaling_5qubit_run/
      (same structure, 5-qubit run)

v1_original/           - original submission's code and paper, unmodified, for reference
```
