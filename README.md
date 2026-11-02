# qembench — a leak-free benchmark for learned quantum error mitigation

Re-evaluates learned error mitigation (random forest / MLP / ridge) against REM and ZNE on
Grover, QAOA and random circuits (4–5 qubits) with disjoint train/test circuits, exact labels,
held-out noise regimes and four evaluation schemes (held-out circuits, leave-one-family-out,
heterogeneous-noise shift, calibrated fake IBM device).

## Quick start
```bash
pip install -r requirements.txt
pip install qiskit-ibm-runtime        # optional: enables the fake-device test (scheme D)
make test                              # unit tests
make quick                             # ~1 min smoke run
make full                              # ~5 min on one core: simulate -> train -> evaluate
make report                            # tables, figures, paper/numbers.tex
make paper                             # compiles paper/main.pdf (needs LaTeX)
```
All numbers in the paper are generated from `results/` via `paper/numbers.tex`.
If you change seeds, versions or settings, **re-read the paper's prose**: qualitative statements
(e.g. "significant in 3 of 4 cells") were checked against the shipped results only.

## Headline findings (shipped results)
- The original "direct" regression design is significantly worse than REM+ZNE almost everywhere.
- Residual learning + XOR canonicalisation gives a small, significant gain on held-out circuits
  and under synthetic noise shift, but **not** across circuit families (B) and **not** on the
  calibrated fake device (D). This is a benchmark/negative-result paper, not a "ML beats ZNE" paper.

## Layout
`qembench/` library · `scripts/` entry points · `tests/` · `results/` CSVs · `paper/` LaTeX ·
`audit/` literature-audit protocol (template only — not yet performed).

## Before you submit (checklist)
1. Verify every reference in `paper/references.bib` against the publisher record.
2. Swap `paper/main.tex` into the target journal's template.
3. Add a data/code availability statement with a Zenodo DOI (enable Zenodo–GitHub integration, tag a release).
4. Optional but valuable: run a few circuits on real IBM hardware (free Open Plan) and perform the audit in `audit/`.
5. Do not describe results beyond what the tables show; keep the limitations section.
