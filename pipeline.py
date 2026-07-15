"""
Shared dataset construction + evaluation logic for the v2 experiments.

Fixes relative to v1 (see config.py / vqe.py / ml_models.py docstrings for
details on each individual fix):
  - Training labels vary (multiple marked states), so the RF/MLP ensemble
    cannot trivially memorize a constant target vector.
  - Every simulator call uses an explicit, varying seed, so shot noise is
    genuinely different across training samples and across repeated
    evaluation trials.
  - The adaptive ensemble weight no longer depends on the true ideal label.
  - The VQE feature is computed under the matching noise model, so it
    actually correlates with noise strength instead of being a constant.
"""
import warnings
import numpy as np

from grover import create_grover
from noise import get_noise
from sim import run_vector, zne
from vqe import run_vqe
from metrics import fidelity, kl, mse
from ml_models import train_rf, train_mlp, ensemble_predict_no_leak
from dataset import normalize

warnings.filterwarnings("ignore")  # quiet repeated qiskit_aer noise-compose warnings


def _vqe_cache(noise_levels, seed_offset=0):
    cache = {}
    for lvl in noise_levels:
        cache[round(float(lvl), 4)] = run_vqe(float(lvl), seed=seed_offset + int(lvl * 1000))
    return cache


def build_dataset(targets, noise_levels, reps, n_qubits, seed_base=0, vqe_cache=None):
    """Returns X (with VQE feature as the last column), y, and the ideal
    vector for every target (needed by the evaluation code)."""
    if vqe_cache is None:
        vqe_cache = _vqe_cache(noise_levels, seed_offset=seed_base)

    ideal_by_target = {}
    X, y = [], []
    seed = seed_base * 100000
    for target in targets:
        qc = create_grover(n_qubits, target)
        ideal_vec = run_vector(qc, n_qubits, noise=None, seed=seed)
        seed += 1
        ideal_by_target[target] = ideal_vec

        for lvl in noise_levels:
            lvl_key = round(float(lvl), 4)
            vqe_val = vqe_cache[lvl_key]
            noise_model = get_noise(float(lvl))
            for _ in range(reps):
                noisy_vec = run_vector(qc, n_qubits, noise_model, seed=seed)
                seed += 1
                X.append(np.concatenate([noisy_vec, [lvl, vqe_val]]))
                y.append(ideal_vec)

    return np.array(X), np.array(y), ideal_by_target, vqe_cache


def predict_variant(rf, nn, feat, mode):
    if mode == "rf_only":
        out = np.maximum(rf.predict([feat])[0], 0)
        return out / out.sum()
    if mode == "mlp_only":
        out = np.maximum(nn.predict([feat])[0], 0)
        return out / out.sum()
    if mode == "ensemble":
        return ensemble_predict_no_leak(rf, nn, feat)
    raise ValueError(mode)


def evaluate_point(target, noise_level, n_qubits, models, use_vqe_feature, vqe_cache,
                    seed, ideal_vec, mode="ensemble"):
    """Runs one noisy trial + ZNE + hybrid-model correction for a single
    (target, noise_level, seed) and returns a dict of metrics for each of
    noisy / zne / hybrid. `mode` selects which trained model(s) to use for
    the "hybrid" prediction (ensemble / rf_only / mlp_only), enabling the
    ablation study to reuse this exact evaluation loop."""
    qc = create_grover(n_qubits, target)
    noise_model = get_noise(noise_level)

    noisy_vec = run_vector(qc, n_qubits, noise_model, seed=seed)
    zne_vec = normalize(zne(qc, n_qubits, seed=seed + 1))

    lvl_key = round(float(noise_level), 4)
    vqe_val = vqe_cache.get(lvl_key)
    if vqe_val is None:
        vqe_val = run_vqe(noise_level, seed=seed)

    if use_vqe_feature:
        feat = np.concatenate([noisy_vec, [noise_level, vqe_val]])
    else:
        feat = np.concatenate([noisy_vec, [noise_level]])

    rf, nn = models
    hybrid_vec = predict_variant(rf, nn, feat, mode=mode)

    out = {}
    for name, vec in [("noisy", noisy_vec), ("zne", zne_vec), ("hybrid", hybrid_vec)]:
        out[name] = {
            "fidelity": fidelity(ideal_vec, vec),
            "kl": kl(ideal_vec, vec),
            "mse": mse(ideal_vec, vec),
        }
    return out



def train_models_on(X, y, use_vqe_feature, seed=42):
    Xp = X if use_vqe_feature else X[:, :-1]
    rf = train_rf(Xp, y, seed=seed)
    nn = train_mlp(Xp, y, seed=seed)
    return rf, nn
