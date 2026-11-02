# Literature audit protocol (to be completed by the authors)

Goal: measure how often published learned-error-mitigation studies would be vulnerable to the
three defects documented in the paper (label degeneracy, label leakage, noise-independent feature)
and to missing baselines. **Every row must be filled in by reading the actual paper. Do not guess.**

## Search
1. Databases: Google Scholar, arXiv (quant-ph, cs.LG), Scopus. Record the date of each search.
2. Queries (examples): "machine learning error mitigation quantum", "learned quantum error mitigation",
   "neural network error mitigation noisy quantum circuits", "Clifford data regression", "regression error mitigation".
3. Inclusion: peer-reviewed or arXiv papers, 2018--present, that train a supervised model to map noisy
   outputs/expectation values to ideal ones. Exclusion: pure ZNE/PEC papers with no learned component.
4. Two reviewers screen independently; resolve disagreements by discussion; report the count at each step (PRISMA-style).

## Coding sheet (`audit_template.csv`)
One row per paper. Code each field as `yes` / `no` / `unclear` (+ page/section where you found it):
- `train_test_circuits_disjoint`: are test circuits (not just noise draws) absent from training?
- `labels_nondegenerate`: do training labels vary across training samples?
- `inference_uses_ground_truth`: does anything at inference time use the ideal result?
- `noise_features_vary`: do all input features actually vary with the noise they claim to capture?
- `seeds_repeated`: results over multiple seeds with variance reported?
- `baselines`: which of ZNE / PEC / REM / CDR / none are compared?
- `hardware`: simulator only / calibrated fake backend / real device?
- `code_available`: yes/no

Report only counts you verified. Put the filled CSV in the repo and cite it in the paper's Section on related practice.
