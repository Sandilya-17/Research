# Hybrid ML+VQE Grover Error Mitigation — v2 (revised per review)

**Start here: [`V2_FINDINGS.md`](V2_FINDINGS.md)** — it explains what was
fixed, why, and what the corrected experiments found (short version: the
original headline result does not survive a genuine held-out test; ZNE
alone outperforms the learned pipeline once the label-degeneracy bug is
fixed).

## Paper

*Label Degeneracy and Label Leakage in Learned Quantum Error Mitigation: A Corrected Evaluation of a Hybrid ML+VQE Pipeline for Grover Search* — full text in [`paper/Label_Degeneracy_Leakage_Learned_QEM_v2.pdf`](paper/Label_Degeneracy_Leakage_Learned_QEM_v2.pdf). Citation: see [`CITATION.cff`](CITATION.cff).

## Headline result (4 qubits, λ = 0.10, mean ± sd over 3 held-out targets × 20 seeds)

| Method | Fidelity ↑ |
|---|---|
| Noisy (no mitigation) | 0.758 ± 0.026 |
| ZNE | 0.879 ± 0.017 |
| Best learned variant (ensemble, no VQE) | 0.690 ± 0.005 |

## Where each paper result comes from

| Paper item | File |
|---|---|
| Table 1 (4-qubit pooled) | `results/main_4qubit_run/summary_pooled_mean_std.csv` |
| Table 2 (per-target) | `results/main_4qubit_run/summary_per_target_mean_std.csv` |
| Figure 2 | `results/main_4qubit_run/pooled_comparison.png` |
| Table 3 (5-qubit) | `results/scaling_5qubit_run/summary_pooled_mean_std.csv` |
| Figure 3 | `results/scaling_5qubit_run/pooled_comparison.png` |
| Appendix A configs | `results/*/config_used.json` |
| Raw trials | `raw_eval_records.csv` in each folder (load with `dtype={"target": str}`) |

The `run_v2_20260715_*` and `run_scaling_v2_20260715_*` folders are not cited in the paper.

## Contents

- `V2_FINDINGS.md` — full write-up of the four requested fixes and results.
- `config.py`, `grover.py`, `noise.py`, `sim.py`, `vqe.py`, `dataset.py`,
  `metrics.py`, `ml_models.py`, `pipeline.py` — the revised pipeline.
- `main_v2.py` — run the 4-qubit generalization + statistics + ablation
  experiment.
- `scaling_v2.py` — run the 5-qubit scaling check.
- `results/main_4qubit_run/`, `results/scaling_5qubit_run/` — saved outputs
  from the runs referenced in `V2_FINDINGS.md` (raw per-trial CSVs,
  aggregated mean±std tables, plots, exact configs used).
- `v1_original/` — the original submitted code and paper, unmodified, kept
  for side-by-side comparison.
- `requirements.txt` — Python dependencies to reproduce the v2 runs.

## Reproducing

```bash
pip install -r requirements.txt
python3 main_v2.py       # ~2-3 minutes; 4-qubit generalization/ablation experiment
python3 scaling_v2.py    # ~1 minute; 5-qubit scaling check
```

Both scripts print a summary table to stdout and write a timestamped
results folder under `results/`.