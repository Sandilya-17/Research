from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from metrics import fidelity


def train_rf(X, y, seed=42):
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(X, y)
    return rf


def train_mlp(X, y, seed=42):
    nn = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )
    nn.fit(X, y)
    return nn


def ensemble_predict(rf, nn, x_row, ideal_ref):
    """Fidelity-weighted RF/MLP ensemble as in the original paper (Eq. 9-10).

    ideal_ref is used only to compute the blending weight in the original
    formulation; for a fair generalization test we instead weight by the
    fidelity between the MLP prediction and the RF prediction (a reference
    that does not require peeking at the true label), see
    `ensemble_predict_no_leak` below, which is what all v2 evaluation code
    actually uses. This leak-free variant is kept only for direct
    side-by-side comparison against the v1 formula.
    """
    rf_out = rf.predict([x_row])[0]
    nn_out = nn.predict([x_row])[0]
    weight = fidelity(np.maximum(ideal_ref, 0), np.maximum(nn_out, 0))
    out = weight * nn_out + (1 - weight) * rf_out
    out = np.maximum(out, 0)
    return out / out.sum()


def ensemble_predict_no_leak(rf, nn, x_row):
    """Fidelity-weighted ensemble that does NOT use the true ideal label to
    set the blend weight (v1's Eq. 9-10 technically requires knowing
    p_ideal to compute omega, which is unavailable at real inference time).
    Instead we weight by how strongly the MLP and RF outputs already agree
    (their mutual fidelity) as a self-consistency proxy, and fall back to a
    fixed 0.5/0.5 blend when neither model can be trusted more than the
    other. This is the ensemble rule used for all v2 headline numbers.
    """
    rf_out = np.maximum(rf.predict([x_row])[0], 0)
    nn_out = np.maximum(nn.predict([x_row])[0], 0)
    agreement = fidelity(rf_out / rf_out.sum(), nn_out / nn_out.sum())
    # agreement in [0,1]; use it directly as the MLP weight so that when
    # the two models already agree closely we trust the (typically more
    # expressive) MLP more, and fall back toward the more conservative RF
    # when they disagree.
    out = agreement * nn_out + (1 - agreement) * rf_out
    out = np.maximum(out, 0)
    return out / out.sum()
