"""Noise models. 'uniform' = i.i.d. Pauli-like noise; 'hetero' = device-like mismatch.

The 'hetero' profile (per-qubit / per-edge rate spread + coherent sx over-rotation) is
used only at *test* time to measure robustness of learned models to model mismatch.
"""
import numpy as np
from qiskit_aer.noise import (NoiseModel, depolarizing_error, amplitude_damping_error,
                              ReadoutError, coherent_unitary_error)
from qiskit.circuit.library import RXGate
from .config import BASIS_GATES


def _clip(p, hi):
    return float(min(max(p, 0.0), hi))


def build_noise_model(profile, scale, n, base, seed=0):
    nm = NoiseModel(basis_gates=BASIS_GATES)
    rng = np.random.default_rng([seed, 7, n])
    if profile == "uniform":
        m1 = np.ones(n); m2 = np.ones((n, n)); mro = np.ones(n); coh = 0.0
    elif profile == "hetero":
        ln = lambda size: np.exp(rng.normal(0, 0.5, size))
        m1 = ln(n); m1 /= m1.mean()
        m2 = ln((n, n)); m2 = (m2 + m2.T) / 2; m2 /= m2.mean()
        mro = ln(n); mro /= mro.mean()
        coh = base["coh"] * scale
    else:
        raise ValueError(profile)

    for q in range(n):
        e1 = depolarizing_error(_clip(base["p1"] * scale * m1[q], 0.5), 1)
        e1 = e1.compose(amplitude_damping_error(_clip(base["amp"] * scale, 0.5)))
        if coh:
            e1 = e1.compose(coherent_unitary_error(RXGate(coh).to_matrix()))
        nm.add_quantum_error(e1, ["sx", "x"], [q])
        r01 = _clip(base["ro01"] * scale * mro[q], 0.45)
        r10 = _clip(base["ro10"] * scale * mro[q], 0.45)
        nm.add_readout_error(ReadoutError([[1 - r01, r01], [r10, 1 - r10]]), [q])
    for a in range(n):
        for b in range(n):
            if a != b:
                e2 = depolarizing_error(_clip(base["p2"] * scale * m2[a, b], 0.9), 2)
                nm.add_quantum_error(e2, ["cx"], [a, b])
    return nm
