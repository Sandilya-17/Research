"""Central experiment configuration. Every number used in the paper comes from here."""
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict

BASIS_GATES = ["rz", "sx", "x", "cx"]


@dataclass
class Config:
    seed: int = 2026
    shots: int = 4096
    cal_shots: int = 16384
    fold_factors: Tuple[int, ...] = (1, 3, 5)          # CNOT-folding noise amplification
    qubit_sizes: Tuple[int, ...] = (4, 5)
    families: Tuple[str, ...] = ("grover", "qaoa", "random")

    # Base per-gate noise at scale 1.0 (roughly "good superconducting device").
    base_noise: Dict[str, float] = field(default_factory=lambda: dict(
        p1=0.002,       # 1q depolarizing on sx / x
        p2=0.025,       # 2q depolarizing on cx
        amp=0.002,      # 1q amplitude damping on sx / x
        ro01=0.020,     # P(read 1 | prepared 0)
        ro10=0.040,     # P(read 0 | prepared 1)
        coh=0.02,       # coherent sx over-rotation (rad), heterogeneous profile only
    ))
    train_scales: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    test_scales_interp: Tuple[float, ...] = (0.75, 1.25)   # between training points
    test_scales_extrap: Tuple[float, ...] = (2.5,)         # beyond training range

    # circuits per family: (n_train, n_test). Grover is limited by the number of marked states.
    n_circuits: Dict[str, Tuple[int, int]] = field(default_factory=lambda: dict(
        qaoa=(40, 20), random=(40, 20)))
    fake_backend: str = "FakeManilaV2"   # calibrated device snapshot from qiskit-ibm-runtime
    reps_train: int = 1
    reps_test: int = 2
    n_boot: int = 10000

    def to_dict(self):
        return asdict(self)


def quick_config() -> Config:
    """Tiny configuration for smoke tests / CI (runs in about a minute)."""
    c = Config()
    c.qubit_sizes = (4,)
    c.shots = 1024
    c.cal_shots = 4096
    c.train_scales = (0.5, 1.5)
    c.test_scales_interp = (1.0,)
    c.test_scales_extrap = (2.5,)
    c.n_circuits = dict(qaoa=(8, 6), random=(8, 6))
    c.reps_test = 1
    c.n_boot = 500
    return c
