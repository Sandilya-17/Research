import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile
from noise import get_noise
from dataset import counts_to_vector
from config import SHOTS, SEED

def run(qc, noise=None):
    sim = AerSimulator(noise_model=noise)
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=SHOTS, seed_simulator=SEED).result()
    return result.get_counts()

def zne(qc):
    levels = [0.05, 0.1, 0.15]
    results = []

    for l in levels:
        counts = run(qc, get_noise(l))
        results.append(counts_to_vector(counts, SHOTS))

    results = np.array(results)
    return results[0] + (results[0] - results[1])