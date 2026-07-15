"""
In v1, run_vqe() used a plain (noiseless) Estimator, so the "VQE hardware-
fidelity proxy" feature was a single constant value regardless of the noise
level being tested -- it could not have carried any information about
device noise, contrary to the paper's framing of it as a physically
motivated, noise-sensitive feature. Here the VQE probe is executed under
the *same* noise model as the Grover circuit at each noise level, so it
actually varies with lambda and can be meaningfully ablated (on vs off).

Implemented directly against qiskit_aer's EstimatorV2 + scipy COBYLA
(rather than qiskit_algorithms.VQE) to avoid a primitives-version
mismatch between qiskit-algorithms (still EstimatorV1-oriented) and the
current qiskit-aer release.

VQE is somewhat expensive to re-run per sample, so callers should cache
run_vqe(level) per distinct noise level (there are only a handful of
distinct levels in the training/eval grids), not call it per shot sample.
"""
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

from noise import get_noise

_HAMILTONIAN = SparsePauliOp.from_list([("ZZ", 1.0), ("IZ", 1.0)])
_ANSATZ = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2).decompose()


def run_vqe(noise_level: float = 0.0, seed: int = 0, maxiter: int = 50) -> float:
    n_params = _ANSATZ.num_parameters

    backend_options = {"seed_simulator": seed}
    if noise_level and noise_level > 0:
        backend_options["noise_model"] = get_noise(noise_level)

    estimator = EstimatorV2(
        options={"backend_options": backend_options, "run_options": {"shots": 1024}}
    )

    def cost(theta):
        job = estimator.run([(_ANSATZ, _HAMILTONIAN, theta)])
        return float(job.result()[0].data.evs)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, size=n_params)
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return float(result.fun)
