"""Conventional (non-learned) baselines: readout-error mitigation (REM) and ZNE variants."""
import numpy as np
from .metrics import normalize


def _apply_1q(p, M, q, n):
    t = p.reshape([2] * n)
    ax = n - 1 - q                      # qubit 0 is the least-significant (last) axis
    t = np.moveaxis(np.tensordot(M, np.moveaxis(t, ax, 0), axes=([1], [0])), 0, ax)
    return t.reshape(-1)


def apply_rem(p, A):
    """Invert tensored assignment matrices A[q] on probability vector p (may go negative)."""
    n = A.shape[0]
    out = np.asarray(p, dtype=float)
    for q in range(n):
        out = _apply_1q(out, np.linalg.inv(A[q]), q, n)
    return out


def zne_weights(xs, deg):
    """Weights w such that sum_i w_i f(x_i) = least-squares polynomial of degree `deg` at x=0.
    deg = len(xs)-1 gives Richardson extrapolation."""
    xs = np.asarray(xs, dtype=float)
    V = np.vander(xs, deg + 1, increasing=True)          # columns: 1, x, x^2, ...
    return np.linalg.pinv(V)[0]                          # row for the intercept


def zne(prob_folds, xs, deg):
    w = zne_weights(xs, deg)
    return np.tensordot(w, prob_folds, axes=(0, 0))


BASELINES = ["raw", "rem", "zne_lin13", "zne_lin135", "zne_rich",
             "rem_zne_lin13", "rem_zne_lin135", "rem_zne_rich"]


def baseline_outputs(probs, A, xs=(1, 3, 5)):
    """probs: (n_folds, N) measured distributions at fold factors xs. Returns dict name->vector."""
    xs = list(xs)
    rem = np.stack([apply_rem(p, A) for p in probs])
    out = {
        "raw": normalize(probs[0]),
        "rem": normalize(rem[0]),
        "zne_lin13": normalize(zne(probs[:2], xs[:2], 1)),
        "zne_lin135": normalize(zne(probs, xs, 1)),
        "zne_rich": normalize(zne(probs, xs, 2)),
        "rem_zne_lin13": normalize(zne(rem[:2], xs[:2], 1)),
        "rem_zne_lin135": normalize(zne(rem, xs, 1)),
        "rem_zne_rich": normalize(zne(rem, xs, 2)),
    }
    return out, np.stack([normalize(r) for r in rem])
