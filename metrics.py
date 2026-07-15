import numpy as np


def kl(p, q):
    return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


def fidelity(p, q):
    return float((np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0)))) ** 2)


def mse(p, q):
    return float(((p - q) ** 2).mean())
