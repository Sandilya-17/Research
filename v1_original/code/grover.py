from qiskit import QuantumCircuit

def create_grover_4(target="1011"):
    qc = QuantumCircuit(4, 4)

    qc.h(range(4))

    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)

    qc.h(3)
    qc.mcx([0,1,2], 3)
    qc.h(3)

    for i, bit in enumerate(target):
        if bit == '0':
            qc.x(i)

    qc.h(range(4))
    qc.x(range(4))

    qc.h(3)
    qc.mcx([0,1,2], 3)
    qc.h(3)

    qc.x(range(4))
    qc.h(range(4))

    qc.measure(range(4), range(4))
    return qc