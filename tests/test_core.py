import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qiskit.quantum_info import Operator, Statevector
from qembench import circuits as C
from qembench.config import Config, quick_config
from qembench.simulate import fold_two_qubit_gates
from qembench.mitigation import apply_rem, zne_weights, zne
from qembench.metrics import hellinger_fidelity, normalize
from qembench.learned import _perm
from qembench.stats import holm


def test_ideal_distribution_normalised_and_grover_peaks_at_target():
    qc = C.grover_circuit(4, 11)
    p = C.ideal_probs(qc)
    assert abs(p.sum() - 1) < 1e-9 and int(np.argmax(p)) == 11
    assert abs(p[11] - np.sin(3 * np.arcsin(0.25)) ** 2) < 1e-9


def test_labels_are_not_degenerate():
    cfg = quick_config()
    labels = [C.ideal_probs(i["circuit"]) for fam in cfg.families
              for i in C.build_instances(fam, 4, "train", cfg)]
    assert len(np.unique(np.round(labels, 10), axis=0)) == len(labels)   # v1 had exactly 1


def test_train_test_targets_and_circuits_disjoint():
    tr, te = C.grover_targets(5, 2026)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == 32
    cfg = quick_config()
    a = C.build_instances("qaoa", 4, "train", cfg); b = C.build_instances("qaoa", 4, "test", cfg)
    pa = {tuple(np.round(C.ideal_probs(x["circuit"]), 8)) for x in a}
    pb = {tuple(np.round(C.ideal_probs(x["circuit"]), 8)) for x in b}
    assert pa.isdisjoint(pb)


def test_folding_preserves_unitary():
    qc = C.compile_circuit(C.qaoa_circuit(3, np.random.default_rng(1)))
    qc.remove_final_measurements()
    for k in (3, 5):
        f = fold_two_qubit_gates(qc, k)
        assert f.count_ops()["cx"] == k * qc.count_ops()["cx"]
        assert Operator(f).equiv(Operator(qc))


def test_zne_recovers_polynomial_exactly():
    xs = [1, 3, 5]
    f = lambda x: 0.7 + 0.05 * x - 0.002 * x ** 2
    vals = np.array([[f(x)] for x in xs])
    assert abs(zne(vals, xs, 2)[0] - f(0)) < 1e-12
    assert abs(zne_weights(xs, 2).sum() - 1) < 1e-12


def test_rem_inverts_tensored_confusion():
    n = 3
    A = np.array([[[0.97, 0.05], [0.03, 0.95]]] * n)
    p = normalize(np.random.default_rng(0).random(2 ** n))
    t = p.reshape([2] * n)
    for q in range(n):
        ax = n - 1 - q
        t = np.moveaxis(np.tensordot(A[q], np.moveaxis(t, ax, 0), axes=([1], [0])), 0, ax)
    assert np.allclose(apply_rem(t.reshape(-1), A), p, atol=1e-10)


def test_xor_canonicalisation_is_involution():
    N = 16
    v = np.random.default_rng(0).random(N)
    for c in range(N):
        pm = _perm(N, c)
        w = v[pm]; back = np.empty_like(w); back[pm] = w
        assert np.allclose(back, v) and np.allclose(w[pm], v)


def test_holm_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    a = holm(p)
    assert np.all(a >= p) and np.all(a <= 1) and np.all(np.diff(a[np.argsort(p)]) >= 0)


def test_metrics():
    p = np.array([0.5, 0.5, 0, 0])
    assert abs(hellinger_fidelity(p, p) - 1) < 1e-12
