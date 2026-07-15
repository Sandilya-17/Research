"""
Global configuration for the revised hybrid ML+VQE Grover error-mitigation
pipeline (v2).

v2 changes vs. the original submission, driven directly by reviewer
concerns:

  1. Training labels are no longer a single fixed ideal distribution.
     TRAIN_TARGETS / TEST_TARGETS below span multiple distinct marked
     states, so y varies across samples and models must learn an actual
     noise-correction map instead of a constant.

  2. Evaluation is repeated over N_EVAL_SEEDS independent shot-noise
     realizations at the held-out noise level, so headline numbers are
     reported as mean +/- std rather than a single-seed point estimate.

  3. Four model configurations are trained and compared on the identical
     held-out set (RF-only, MLP-only, ensemble-without-VQE-feature,
     full ensemble-with-VQE-feature) to isolate each component's
     contribution.

  4. A second, larger circuit size (N_QUBITS_SCALING qubits) is run at
     reduced sample density to honestly characterize how performance
     degrades as the O(2^n) label space grows, supporting the scalability
     discussion.
"""
import numpy as np
import random

SEED = 42
SHOTS = 2048

# ---- 4-qubit primary experiment ------------------------------------------
N_QUBITS = 4
ALL_4BIT = [format(i, "04b") for i in range(16)]

# Targets used to build the *training* corpus (labels differ across these).
TRAIN_TARGETS = ["0000", "0011", "0101", "0110", "1001", "1010", "1100", "1111",
                  "0001", "1110"]
# Targets held out completely from training -> used for the generalization /
# statistical-robustness evaluation (addresses reviewer point 1 and 2).
TEST_TARGETS = ["1011", "0100", "1101"]

NOISE_LEVELS_TRAIN = np.round(np.linspace(0.02, 0.18, 6), 4)
REPS_PER_CELL = 15          # noisy/ideal execution pairs per (target, noise)
TEST_NOISE_LEVEL = 0.10     # headline evaluation noise level (interpolated,
                            # not one of the 6 training grid points)
N_EVAL_SEEDS = 20           # independent shot-noise repetitions per test target

# ---- Scaling experiment (5 qubits, 32 outcomes) ---------------------------
N_QUBITS_SCALING = 5
SCALING_TRAIN_TARGETS = ["00000", "01010", "10101", "11111", "00110", "11001"]
SCALING_TEST_TARGETS = ["10011", "01101"]
SCALING_NOISE_LEVELS_TRAIN = np.round(np.linspace(0.02, 0.18, 4), 4)
SCALING_REPS_PER_CELL = 10
SCALING_TEST_NOISE_LEVEL = 0.10
SCALING_N_EVAL_SEEDS = 10


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
