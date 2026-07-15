import numpy as np


def counts_to_vector(counts, shots, n_qubits):
    n_states = 2 ** n_qubits
    keys = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    vec = np.array([counts.get(k, 0) for k in keys], dtype=float)
    return vec / shots


def normalize(vec):
    vec = np.maximum(vec, 0)
    total = np.sum(vec)
    if total <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / total
