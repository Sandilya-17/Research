"""
v1 always called the simulator with seed_simulator=SEED (a single fixed
constant), which meant every "ideal" execution of a fixed circuit returned
bit-for-bit identical shot statistics. Combined with the fact that every
training sample used the same circuit/target, this made the ML label
vector literally constant across the whole training set. Here `seed` is a
required, varying argument so we get genuine shot-noise diversity across
training samples and across repeated evaluation trials.
"""
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from dataset import counts_to_vector
from config import SHOTS


def run(qc, noise=None, seed=0):
    sim = AerSimulator(noise_model=noise)
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=SHOTS, seed_simulator=seed).result()
    return result.get_counts()


def run_vector(qc, n_qubits, noise=None, seed=0):
    counts = run(qc, noise=noise, seed=seed)
    return counts_to_vector(counts, SHOTS, n_qubits)


def zne(qc, n_qubits, seed=0):
    """Two-point Richardson extrapolation, as in v1, but with an explicit
    (varying) seed so ZNE outputs also carry realistic shot noise."""
    from noise import get_noise
    levels = [0.05, 0.10, 0.15]
    results = []
    for i, lvl in enumerate(levels):
        vec = run_vector(qc, n_qubits, get_noise(lvl), seed=seed + 1000 * i)
        results.append(vec)
    results = np.array(results)
    return results[0] + (results[0] - results[1])
