"""Paired statistics. Unit of analysis = circuit (scales/reps within a regime are averaged),
so repeated shot-noise draws are never treated as independent samples."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .mitigation import BASELINES
from .learned import LEARNED

CELL = ["scheme", "profile", "n", "regime"]


def circuit_level(df, keys=("scheme", "held", "profile", "n", "family", "regime", "cid", "method")):
    return (df.groupby(list(keys), dropna=False)[["fidelity", "tvd", "tprob"]].mean().reset_index())


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def boot_ci(d, n_boot, rng, alpha=0.05):
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])


def compare(cl, group_cols, n_boot=10000, seed=0, metric="fidelity"):
    """For every cell defined by group_cols: learned vs best baseline (oracle-chosen, i.e. conservative)."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in cl.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        piv = g.pivot_table(index="cid", columns="method", values=metric)
        base_cols = [b for b in BASELINES if b in piv.columns and b != "raw"]
        if not base_cols:
            continue
        best = piv[base_cols].mean().idxmax()
        raw = piv["raw"]
        for m in LEARNED:
            if m not in piv.columns:
                continue
            d = (piv[m] - piv[best]).dropna().values
            if len(d) < 2:
                continue
            try:
                p = wilcoxon(d, alternative="two-sided").pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            lo, hi = boot_ci(d, n_boot, rng)
            F = piv[m].mean(); Fr = raw.mean(); Fb = piv[best].mean()
            rows.append(dict(zip(group_cols, key), method=m, best_baseline=best, n_circuits=len(d),
                             F_method=F, F_best=Fb, F_raw=Fr, diff=d.mean(), ci_lo=lo, ci_hi=hi,
                             p=p, win_rate=(d > 0).mean(),
                             infid_red_vs_raw=1 - (1 - F) / (1 - Fr) if Fr < 1 else np.nan,
                             infid_red_best_vs_raw=1 - (1 - Fb) / (1 - Fr) if Fr < 1 else np.nan))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].values)
    return out
