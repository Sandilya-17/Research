from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error


def get_noise(level):
    noise_model = NoiseModel()

    dep = depolarizing_error(level, 1)
    amp = amplitude_damping_error(level)

    noise_model.add_all_qubit_quantum_error(dep, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(amp, ['u1', 'u2', 'u3'])

    return noise_model
