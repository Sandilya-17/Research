"""Noisy execution with CNOT folding, readout calibration and on-disk caching."""
import os, pickle, time, zlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from .config import BASIS_GATES
from .noise import build_noise_model
from . import circuits as C

SELF_INVERSE_2Q = {"cx", "cz", "ecr"}


def fold_two_qubit_gates(qc, k):
    """Replace every self-inverse 2q gate G by G (G G)^((k-1)/2); k odd. Unitary preserved."""
    assert k % 2 == 1 and k >= 1
    if k == 1:
        return qc
    out = qc.copy_empty_like()
    for inst in qc.data:
        out.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name in SELF_INVERSE_2Q:
            for _ in range(k - 1):
                out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def _seed(*parts):
    return int(np.random.default_rng(list(parts)).integers(1, 2**31 - 1))


def _counts_to_probs(counts, n, shots):
    p = np.zeros(2 ** n)
    for k, v in counts.items():
        p[int(k.replace(" ", ""), 2)] += v
    return p / shots


def run_probs(circ, nm, shots, seed):
    sim = AerSimulator(noise_model=nm, method="density_matrix", fusion_enable=False,
                       basis_gates=BASIS_GATES)
    res = sim.run(circ, shots=shots, seed_simulator=seed).result()
    return _counts_to_probs(res.get_counts(), circ.num_qubits, shots)


def calibrate_readout(n, nm, shots, seed):
    """Tensored per-qubit assignment matrices A[q] = [[P(0|0),P(0|1)],[P(1|0),P(1|1)]]."""
    c0 = QuantumCircuit(n); c0.measure_all()
    c1 = QuantumCircuit(n); c1.x(range(n)); c1.measure_all()
    p0 = run_probs(c0, nm, shots, seed)
    p1 = run_probs(c1, nm, shots, seed + 1)
    A = np.zeros((n, 2, 2))
    for q in range(n):
        idx = np.arange(2 ** n)
        bit = (idx >> q) & 1
        m0 = p0[bit == 1].sum(); m1 = p1[bit == 0].sum()   # P(1|0), P(0|1)
        A[q] = [[1 - m0, m1], [m0, 1 - m1]]
    return A


def simulate_group(cfg, profile, n, family, splits=("train", "test"), verbose=True):
    """Return list of record dicts for one (profile, n, family)."""
    records = []
    cal_cache = {}
    for split in splits:
        insts = C.build_instances(family, n, split, cfg)
        if split == "train":
            scales, reps = list(cfg.train_scales), cfg.reps_train
        else:
            scales = list(cfg.train_scales) + list(cfg.test_scales_interp) + list(cfg.test_scales_extrap)
            reps = cfg.reps_test
        for inst in insts:
            ideal = C.ideal_probs(inst["circuit"])
            comp = C.compile_circuit(inst["circuit"], seed=cfg.seed)
            ncx = C.count_cx(comp)
            folded = {k: fold_two_qubit_gates(comp, k) for k in cfg.fold_factors}
            for s in scales:
                if s not in cal_cache:
                    nm0 = build_noise_model(profile, s, n, cfg.base_noise, cfg.seed)
                    cal_cache[s] = (nm0, calibrate_readout(n, nm0, cfg.cal_shots, _seed(cfg.seed, 1, n, int(s * 1000))))
                nm, A = cal_cache[s]
                regime = ("train" if s in cfg.train_scales else
                          "interp" if s in cfg.test_scales_interp else "extrap")
                for r in range(reps):
                    probs = np.stack([
                        run_probs(folded[k], nm, cfg.shots,
                                  _seed(cfg.seed, 2, n, zlib.crc32(inst["cid"].encode()) % 100003, int(s * 1000), k, r))
                        for k in cfg.fold_factors])
                    records.append(dict(profile=profile, n=n, family=family, split=split,
                                        cid=inst["cid"], target=inst["target"], scale=float(s),
                                        regime=regime, rep=r, ncx=ncx, depth=comp.depth(),
                                        probs=probs, ideal=ideal, A=A))
    return records


def _load_fake_backend(name):
    try:
        from qiskit_ibm_runtime import fake_provider as fp
        return getattr(fp, name)()
    except Exception as e:                      # optional dependency
        print(f"[warn] fake backend {name} unavailable ({type(e).__name__}); skipping device test")
        return None


def simulate_fakedev(cfg, n, family):
    """Test circuits on a calibrated fake device (real calibration snapshot: T1/T2, gate and readout
    errors, linear connectivity, SWAP routing). Used only as a *test* distribution (scheme D)."""
    from qiskit import transpile
    be = _load_fake_backend(cfg.fake_backend)
    if be is None or be.num_qubits < n:
        return []
    sim = AerSimulator.from_backend(be, method="density_matrix")
    layout = list(range(n))

    def tr(c):
        return transpile(c, backend=be, optimization_level=1, initial_layout=layout, seed_transpiler=cfg.seed)

    c0 = QuantumCircuit(n); c0.measure_all()
    c1 = QuantumCircuit(n); c1.x(range(n)); c1.measure_all()
    def runp(c, shots, seed):
        res = sim.run(c, shots=shots, seed_simulator=seed).result()
        return _counts_to_probs(res.get_counts(), n, shots)
    p0 = runp(tr(c0), cfg.cal_shots, _seed(cfg.seed, 5, n)); p1 = runp(tr(c1), cfg.cal_shots, _seed(cfg.seed, 6, n))
    A = np.zeros((n, 2, 2)); idx = np.arange(2 ** n)
    for q in range(n):
        bit = (idx >> q) & 1
        m0 = p0[bit == 1].sum(); m1 = p1[bit == 0].sum()
        A[q] = [[1 - m0, m1], [m0, 1 - m1]]
    records = []
    for inst in C.build_instances(family, n, "test", cfg):
        ideal = C.ideal_probs(inst["circuit"])
        logical = inst["circuit"].copy(); logical.measure_all()
        comp = tr(logical)
        ncx = int(comp.count_ops().get("cx", 0))
        folded = {k: fold_two_qubit_gates(comp, k) for k in cfg.fold_factors}
        for r in range(cfg.reps_test):
            probs = np.stack([runp(folded[k], cfg.shots,
                                   _seed(cfg.seed, 7, n, zlib.crc32(inst["cid"].encode()) % 100003, k, r))
                              for k in cfg.fold_factors])
            records.append(dict(profile="fakedev", n=n, family=family, split="test", cid=inst["cid"],
                                target=inst["target"], scale=1.0, regime="device", rep=r, ncx=ncx,
                                depth=comp.depth(), probs=probs, ideal=ideal, A=A))
    return records


def simulate_all(cfg, outdir, profiles=("uniform", "hetero", "fakedev"), force=False, verbose=True):
    os.makedirs(outdir, exist_ok=True)
    allrec = []
    for profile in profiles:
        splits = ("train", "test") if profile == "uniform" else ("test",)
        for n in cfg.qubit_sizes:
            for fam in cfg.families:
                path = os.path.join(outdir, f"sim_{profile}_{n}_{fam}.pkl")
                if os.path.exists(path) and not force:
                    with open(path, "rb") as f:
                        recs = pickle.load(f)
                else:
                    t0 = time.time()
                    recs = (simulate_fakedev(cfg, n, fam) if profile == "fakedev"
                            else simulate_group(cfg, profile, n, fam, splits, verbose))
                    with open(path, "wb") as f:
                        pickle.dump(recs, f)
                    if verbose:
                        print(f"[sim] {profile:8s} n={n} {fam:7s} {len(recs):5d} records "
                              f"{time.time()-t0:6.1f}s", flush=True)
                allrec.extend(recs)
    return allrec
