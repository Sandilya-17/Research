"""Benchmark circuit families with exact (statevector) ideal output distributions.

Labels are *exact* probabilities, not shot-sampled histograms, so ideal labels are
never degenerate because of a fixed simulator seed (the defect of the earlier pipeline).
Bit ordering follows Qiskit: index i of a distribution <-> bitstring of i, qubit 0 = LSB.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from .config import BASIS_GATES

FAMILY_ID = {"grover": 1, "qaoa": 2, "random": 3}


def _rng(seed, family, n, idx):
    return np.random.default_rng([seed, FAMILY_ID[family], n, idx])


def grover_circuit(n, target, iters=1):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iters):
        # oracle: phase-flip |target>
        zeros = [q for q in range(n) if not (target >> q) & 1]
        if zeros:
            qc.x(zeros)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        if zeros:
            qc.x(zeros)
        # diffusion
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
    return qc


def qaoa_circuit(n, rng, p=2):
    edges = [(i, (i + 1) % n) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if (i, j) != (0, n - 1) and rng.random() < 0.3:
                edges.append((i, j))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        gamma = rng.uniform(0.2, 1.4)
        beta = rng.uniform(0.2, 1.2)
        for (i, j) in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for q in range(n):
            qc.rx(2 * beta, q)
    return qc


def random_circuit(n, rng, depth=8):
    qc = QuantumCircuit(n)
    for layer in range(depth):
        for q in range(n):
            qc.u(rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), q)
        for a in range(layer % 2, n - 1, 2):
            qc.cx(a, a + 1)
    return qc


def grover_targets(n, seed):
    """Disjoint train/test split of marked states (50/50) -> held-out *targets*."""
    rng = np.random.default_rng([seed, 99, n])
    perm = rng.permutation(2 ** n)
    half = 2 ** n // 2
    return sorted(perm[:half].tolist()), sorted(perm[half:].tolist())


def build_instances(family, n, split, cfg):
    """Return list of dicts {cid, circuit, target} for a family / split ('train'|'test')."""
    out = []
    if family == "grover":
        tr, te = grover_targets(n, cfg.seed)
        targets = tr if split == "train" else te
        for k, t in enumerate(targets):
            out.append(dict(cid=f"grover-{n}-{split}-{k}", circuit=grover_circuit(n, t), target=t))
        return out
    n_tr, n_te = cfg.n_circuits[family]
    count = n_tr if split == "train" else n_te
    offset = 0 if split == "train" else 10_000          # disjoint RNG streams
    for k in range(count):
        rng = _rng(cfg.seed, family, n, offset + k)
        qc = qaoa_circuit(n, rng) if family == "qaoa" else random_circuit(n, rng)
        out.append(dict(cid=f"{family}-{n}-{split}-{k}", circuit=qc, target=-1))
    return out


def ideal_probs(qc):
    return np.asarray(Statevector.from_instruction(qc).probabilities(), dtype=float)


def compile_circuit(qc, seed=0):
    """Transpile to the native basis once; measurement is appended afterwards."""
    t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=seed)
    m = t.copy()
    m.measure_all()
    return m


def count_cx(qc):
    return int(qc.count_ops().get("cx", 0))
