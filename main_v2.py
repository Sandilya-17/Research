import os
import json
import datetime
import numpy as np
import pandas as pd

from config import (set_seed, TRAIN_TARGETS, TEST_TARGETS, NOISE_LEVELS_TRAIN,
                     REPS_PER_CELL, TEST_NOISE_LEVEL, N_EVAL_SEEDS, N_QUBITS)
from pipeline import build_dataset, train_models_on, evaluate_point

set_seed()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"results/run_v2_{timestamp}"
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 70)
print("V2 EXPERIMENT: 4-qubit generalization + statistics + ablation")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Build training corpus (labels vary across TRAIN_TARGETS -> fixes the
#    label-degeneracy issue).
# ---------------------------------------------------------------------------
print(f"\nBuilding training set: {len(TRAIN_TARGETS)} targets x "
      f"{len(NOISE_LEVELS_TRAIN)} noise levels x {REPS_PER_CELL} reps ...")
X, y, ideal_by_train_target, vqe_cache = build_dataset(
    TRAIN_TARGETS, NOISE_LEVELS_TRAIN, REPS_PER_CELL, N_QUBITS, seed_base=1
)
print(f"Training set shape: X={X.shape}, y={y.shape}")
print(f"Unique label rows in y: {len(np.unique(y, axis=0))} "
      f"(should equal number of training targets = {len(TRAIN_TARGETS)}; "
      f"in v1 this number was 1 for the entire dataset)")

# Extend the VQE cache so it also covers the held-out test noise level.
from vqe import run_vqe
test_lvl_key = round(float(TEST_NOISE_LEVEL), 4)
if test_lvl_key not in vqe_cache:
    vqe_cache[test_lvl_key] = run_vqe(TEST_NOISE_LEVEL, seed=999)

# ---------------------------------------------------------------------------
# 2. Train the four model configurations used throughout (full ensemble is
#    the "headline" v2 model; the other three feed the ablation table).
# ---------------------------------------------------------------------------
print("\nTraining model variants ...")
rf_nn_with_vqe = train_models_on(X, y, use_vqe_feature=True)
rf_nn_no_vqe = train_models_on(X, y, use_vqe_feature=False)
print("Done training.")

variants = {
    "ensemble_with_VQE": {"models": rf_nn_with_vqe, "use_vqe": True, "mode": "ensemble"},
    "ensemble_no_VQE": {"models": rf_nn_no_vqe, "use_vqe": False, "mode": "ensemble"},
    "RF_only_with_VQE": {"models": rf_nn_with_vqe, "use_vqe": True, "mode": "rf_only"},
    "MLP_only_with_VQE": {"models": rf_nn_with_vqe, "use_vqe": True, "mode": "mlp_only"},
}

# ---------------------------------------------------------------------------
# 3. Generalization + multi-seed evaluation. For every held-out test target
#    (never seen in training) and N_EVAL_SEEDS independent shot-noise
#    realizations at TEST_NOISE_LEVEL, evaluate noisy / ZNE / each model
#    variant. This directly addresses reviewer points 1 and 2.
# ---------------------------------------------------------------------------
print(f"\nEvaluating on {len(TEST_TARGETS)} held-out targets x "
      f"{N_EVAL_SEEDS} seeds at noise level {TEST_NOISE_LEVEL} ...")

records = []
for target in TEST_TARGETS:
    from grover import create_grover
    from sim import run_vector
    ideal_vec = run_vector(create_grover(N_QUBITS, target), N_QUBITS, noise=None, seed=5)

    for eval_seed in range(N_EVAL_SEEDS):
        seed = 500000 + eval_seed * 97 + hash(target) % 1000
        for variant_name, cfg in variants.items():
            result = evaluate_point(
                target, TEST_NOISE_LEVEL, N_QUBITS, cfg["models"], cfg["use_vqe"],
                vqe_cache, seed, ideal_vec, mode=cfg["mode"],
            )
            for stage in ["noisy", "zne", "hybrid"]:
                row = {
                    "target": target, "eval_seed": eval_seed, "variant": variant_name,
                    "stage": stage if stage != "hybrid" else variant_name,
                    "fidelity": result[stage]["fidelity"],
                    "kl": result[stage]["kl"],
                    "mse": result[stage]["mse"],
                }
                records.append(row)

df = pd.DataFrame(records)
df.to_csv(f"{RESULT_DIR}/raw_eval_records.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Aggregate: mean +/- std per (target, stage) and pooled across targets.
# ---------------------------------------------------------------------------
# noisy/zne are duplicated per variant loop (same seed/target); dedupe them.
baseline_df = df[df["stage"].isin(["noisy", "zne"])].drop_duplicates(
    subset=["target", "eval_seed", "stage"]
)
hybrid_df = df[~df["stage"].isin(["noisy", "zne"])]
stats_input = pd.concat([baseline_df, hybrid_df], ignore_index=True)

summary = (
    stats_input.groupby(["target", "stage"])[["fidelity", "kl", "mse"]]
    .agg(["mean", "std"])
)
summary.to_csv(f"{RESULT_DIR}/summary_per_target_mean_std.csv")

pooled = (
    stats_input.groupby("stage")[["fidelity", "kl", "mse"]]
    .agg(["mean", "std"])
)
pooled.to_csv(f"{RESULT_DIR}/summary_pooled_mean_std.csv")

print("\n===== POOLED RESULTS (mean +/- std over held-out targets x seeds) =====")
print(pooled.round(4).to_string())

# ---------------------------------------------------------------------------
# 5. Save everything needed to reproduce / audit.
# ---------------------------------------------------------------------------
config_dump = {
    "train_targets": TRAIN_TARGETS,
    "test_targets": TEST_TARGETS,
    "noise_levels_train": list(NOISE_LEVELS_TRAIN),
    "reps_per_cell": REPS_PER_CELL,
    "test_noise_level": TEST_NOISE_LEVEL,
    "n_eval_seeds": N_EVAL_SEEDS,
    "n_qubits": N_QUBITS,
    "training_set_shape": list(X.shape),
    "unique_label_rows": int(len(np.unique(y, axis=0))),
}
with open(f"{RESULT_DIR}/config_used.json", "w") as f:
    json.dump(config_dump, f, indent=2)

print(f"\nSaved all v2 main-experiment results to: {RESULT_DIR}")
print(RESULT_DIR)
