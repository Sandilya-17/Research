from qiskit import QuantumCircuit


def create_grover(n_qubits: int, target: str) -> QuantumCircuit:
    """Single-iteration Grover circuit on n_qubits, marking `target`.

    Generalizes the original 4-qubit-only implementation so the same
    pipeline can build circuits for different marked states (needed to
    break the label-degeneracy issue) and for a larger qubit count
    (needed for the scaling experiment).
    """
    assert len(target) == n_qubits
    qc = QuantumCircuit(n_qubits, n_qubits)

    qc.h(range(n_qubits))

    # Oracle: phase-flip the target basis state.
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)

    # Diffuser.
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def create_grover_4(target="1011"):
    """Kept for backward compatibility with the original v1 module."""
    return create_grover(4, target)
