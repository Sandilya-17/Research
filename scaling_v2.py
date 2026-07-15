"""
Reviewer point 4 (extend to more qubits / be honest about scale limits).

Repeats the same generalization + multi-seed evaluation protocol as
main_v2.py but at 5 qubits (32-outcome label space) with a training corpus
sized the same way as the 4-qubit run (roughly comparable samples-per-
training-target), to make the O(2^n) scaling cost directly visible rather
than just asserted in prose.
"""
import os
import json
import datetime
import numpy as np
import pandas as pd

from config import (set_seed, SCALING_TRAIN_TARGETS, SCALING_TEST_TARGETS,
                     SCALING_NOISE_LEVELS_TRAIN, SCALING_REPS_PER_CELL,
                     SCALING_TEST_NOISE_LEVEL, SCALING_N_EVAL_SEEDS, N_QUBITS_SCALING)
from pipeline import build_dataset, train_models_on, evaluate_point
from grover import create_grover
from sim import run_vector
from vqe import run_vqe

set_seed()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"results/run_scaling_v2_{timestamp}"
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 70)
print(f"V2 SCALING EXPERIMENT: {N_QUBITS_SCALING} qubits "
      f"({2**N_QUBITS_SCALING} outcomes)")
print("=" * 70)

print(f"\nBuilding training set: {len(SCALING_TRAIN_TARGETS)} targets x "
      f"{len(SCALING_NOISE_LEVELS_TRAIN)} noise levels x {SCALING_REPS_PER_CELL} reps ...")
X, y, ideal_by_target, vqe_cache = build_dataset(
    SCALING_TRAIN_TARGETS, SCALING_NOISE_LEVELS_TRAIN, SCALING_REPS_PER_CELL,
    N_QUBITS_SCALING, seed_base=2,
)
print(f"Training set shape: X={X.shape}, y={y.shape}  "
      f"(label dimensionality = {2**N_QUBITS_SCALING}, "
      f"samples-per-label-dimension = {X.shape[0] / (2**N_QUBITS_SCALING):.2f}; "
      f"compare to the 4-qubit run's samples-per-label-dimension)")

test_lvl_key = round(float(SCALING_TEST_NOISE_LEVEL), 4)
if test_lvl_key not in vqe_cache:
    vqe_cache[test_lvl_key] = run_vqe(SCALING_TEST_NOISE_LEVEL, seed=999)

print("\nTraining ensemble (with VQE feature) ...")
models = train_models_on(X, y, use_vqe_feature=True)

print(f"\nEvaluating on {len(SCALING_TEST_TARGETS)} held-out targets x "
      f"{SCALING_N_EVAL_SEEDS} seeds at noise level {SCALING_TEST_NOISE_LEVEL} ...")

records = []
for target in SCALING_TEST_TARGETS:
    ideal_vec = run_vector(create_grover(N_QUBITS_SCALING, target), N_QUBITS_SCALING,
                            noise=None, seed=5)
    for eval_seed in range(SCALING_N_EVAL_SEEDS):
        seed = 700000 + eval_seed * 131 + hash(target) % 1000
        result = evaluate_point(
            target, SCALING_TEST_NOISE_LEVEL, N_QUBITS_SCALING, models, True,
            vqe_cache, seed, ideal_vec, mode="ensemble",
        )
        for stage in ["noisy", "zne", "hybrid"]:
            records.append({
                "target": target, "eval_seed": eval_seed,
                "stage": "hybrid_ensemble" if stage == "hybrid" else stage,
                "fidelity": result[stage]["fidelity"],
                "kl": result[stage]["kl"],
                "mse": result[stage]["mse"],
            })

df = pd.DataFrame(records)
df.to_csv(f"{RESULT_DIR}/raw_eval_records.csv", index=False)

pooled = df.groupby("stage")[["fidelity", "kl", "mse"]].agg(["mean", "std"])
pooled.to_csv(f"{RESULT_DIR}/summary_pooled_mean_std.csv")

print("\n===== 5-QUBIT POOLED RESULTS (mean +/- std) =====")
print(pooled.round(4).to_string())

config_dump = {
    "n_qubits": N_QUBITS_SCALING,
    "n_outcomes": 2 ** N_QUBITS_SCALING,
    "train_targets": SCALING_TRAIN_TARGETS,
    "test_targets": SCALING_TEST_TARGETS,
    "noise_levels_train": list(SCALING_NOISE_LEVELS_TRAIN),
    "reps_per_cell": SCALING_REPS_PER_CELL,
    "test_noise_level": SCALING_TEST_NOISE_LEVEL,
    "n_eval_seeds": SCALING_N_EVAL_SEEDS,
    "training_set_shape": list(X.shape),
    "samples_per_label_dim": X.shape[0] / (2 ** N_QUBITS_SCALING),
}
with open(f"{RESULT_DIR}/config_used.json", "w") as f:
    json.dump(config_dump, f, indent=2)

print(f"\nSaved all v2 scaling-experiment results to: {RESULT_DIR}")
print(RESULT_DIR)
