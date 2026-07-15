# Hybrid ML+VQE Grover Error Mitigation — v2 (revised per review)

**Start here: [`V2_FINDINGS.md`](V2_FINDINGS.md)** — it explains what was
fixed, why, and what the corrected experiments found (short version: the
original headline result does not survive a genuine held-out test; ZNE
alone outperforms the learned pipeline once the label-degeneracy bug is
fixed).

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
