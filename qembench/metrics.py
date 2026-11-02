import numpy as np


def hellinger_fidelity(p, q):
    return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))) ** 2)


def tvd(p, q):
    return float(0.5 * np.abs(p - q).sum())


def normalize(p):
    """Clip negative quasi-probabilities and renormalise (standard post-processing)."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))
