"""Learned mitigators. All are trained only on (noisy features -> exact ideal) pairs from
*training* circuits; nothing about the test circuit's ideal output is used at inference.

Variants
  direct_*   : replicate the original design (noisy histogram -> ideal histogram).
  res_*      : residual learning on top of REM+ZNE (richardson): ideal = base + f(x).
  *_canon    : XOR-canonicalisation. Bit-strings are re-labelled by XOR with argmax of the
               REM'd noisy histogram, so every circuit's dominant outcome sits at index 0.
               This injects the translation symmetry of the label space and is computed
               from measured data only (no label information).
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from .metrics import normalize
from .mitigation import baseline_outputs

LEARNED = ["direct_rf", "direct_mlp", "res_ridge", "res_rf", "res_mlp",
           "res_ridge_canon", "res_rf_canon", "res_mlp_canon", "mean_canon"]


class Reg:
    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def _make(self):
        if self.kind == "ridge":
            return RidgeCV(alphas=np.logspace(-3, 3, 13))
        if self.kind == "rf":
            return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.5,
                                         random_state=self.seed, n_jobs=1)
        if self.kind == "mlp":
            return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
                                early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=25, random_state=self.seed)
        raise ValueError(self.kind)

    def fit(self, X, y):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ys = y.std() + 1e-12
        self.m = self._make().fit((X - self.xm) / self.xs, y / self.ys)
        return self

    def predict(self, X):
        return self.m.predict((X - self.xm) / self.xs) * self.ys


BASE_CANDIDATES = ["rem_zne_lin13", "rem_zne_lin135", "rem_zne_rich"]


def select_base(recs, preps):
    """Leak-free model selection: pick the REM+ZNE variant with best mean fidelity on TRAINING data."""
    from .metrics import hellinger_fidelity
    scores = {b: np.mean([hellinger_fidelity(d["outs"][b], r["ideal"]) for r, d in zip(recs, preps)])
              for b in BASE_CANDIDATES}
    return max(scores, key=scores.get)


def prepare(rec, xs=(1, 3, 5)):
    """Per-record derived quantities (all available at inference time)."""
    outs, rem_folds = baseline_outputs(rec["probs"], rec["A"], xs)
    n = rec["n"]
    ro = float(np.mean([(rec["A"][q][1, 0] + rec["A"][q][0, 1]) / 2 for q in range(n)]))
    extras = np.array([ro * 10, rec["ncx"] / 100.0, rec["depth"] / 200.0])
    c = int(np.argmax(rem_folds[0]))
    return dict(outs=outs, rem=rem_folds, extras=extras, c=c, base=outs["rem_zne_rich"],
                raw=normalize(rec["probs"][0]))


def _perm(N, c):
    return np.arange(N) ^ c


def build_xy(recs, preps, kind, base_name="rem_zne_rich"):
    """Return X, y (training targets) and helpers for a variant kind."""
    X, Y, B, P = [], [], [], []
    for r, d in zip(recs, preps):
        N = len(r["ideal"]); pm = _perm(N, d["c"]) if kind.endswith("canon") else np.arange(N)
        if kind.startswith("direct"):
            x = np.concatenate([d["raw"], d["extras"]]); base = np.zeros(N)
        else:
            x = np.concatenate([d["rem"][0][pm], d["rem"][1][pm], d["rem"][2][pm], d["extras"]])
            base = d["outs"][base_name][pm]
        X.append(x); Y.append(r["ideal"][pm] - base); B.append(base); P.append(pm)
    return np.array(X), np.array(Y), np.array(B), np.array(P)


def variant_spec(name):
    kind = name.rsplit("_canon", 1)[0] if name.endswith("_canon") else name
    reg = kind.split("_")[1]
    layout = "direct" if name.startswith("direct") else "res"
    canon = name.endswith("canon")
    return layout, reg, canon


def train_all(recs, preps, seed=0, variants=LEARNED):
    base_name = select_base(recs, preps)
    models = {"_base": base_name}
    for v in variants:
        if v == "mean_canon":
            ideal_canon = np.array([r["ideal"][_perm(len(r["ideal"]), d["c"])] for r, d in zip(recs, preps)])
            models[v] = ideal_canon.mean(0)
            continue
        layout, reg, canon = variant_spec(v)
        tag = f"{layout}_x" + ("_canon" if canon else "")
        X, Y, _, _ = build_xy(recs, preps, tag, base_name)
        models[v] = Reg(reg, seed).fit(X, Y)
    return models


def predict_all(models, recs, preps, variants=LEARNED):
    out = {v: [] for v in variants}
    for v in variants:
        if v == "mean_canon":
            for r, d in zip(recs, preps):
                N = len(r["ideal"]); pm = _perm(N, d["c"])
                out[v].append(normalize(models[v][pm]))
            continue
        layout, reg, canon = variant_spec(v)
        tag = f"{layout}_x" + ("_canon" if canon else "")
        X, _, B, P = build_xy(recs, preps, tag, models["_base"])
        pred = models[v].predict(X) + B                  # canonical-frame prediction
        for row, pm in zip(pred, P):
            res = np.empty_like(row); res[pm] = row      # undo permutation (XOR is an involution)
            out[v].append(normalize(res))
    return out
