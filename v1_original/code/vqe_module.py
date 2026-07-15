from qiskit_aer import AerSimulator
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

def run_vqe():
    ansatz = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("IZ", 1.0)
    ])

    optimizer = COBYLA(maxiter=50)

    estimator = Estimator()

    vqe = VQE(estimator, ansatz, optimizer)

    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    return result.eigenvalue.real