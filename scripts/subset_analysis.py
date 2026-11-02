#!/usr/bin/env python3
"""Post hoc subset analysis (paper Sec. 4.4): Grover-only vs QAOA+random-only vs all.
Same procedure as the pooled analysis (best baseline chosen within each subset, paired
differences over circuits, bootstrap CI, Wilcoxon, Holm over all comparisons in the subset).
Writes results/stats_subsets.csv and prints the res_rf_canon rows (paper Table 4)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from qembench.stats import circuit_level, compare

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
ap.add_argument("--n_boot", type=int, default=10000); a = ap.parse_args()
df = pd.read_csv(os.path.join(a.results, "raw_results.csv"))
cl = circuit_level(df)
cl = cl[cl["regime"] != "train"]
subsets = {"all": cl, "grover": cl[cl.family == "grover"], "qaoa_random": cl[cl.family != "grover"]}
out = []
for name, sub in subsets.items():
    r = compare(sub, ["scheme", "profile", "n", "regime"], a.n_boot); r.insert(0, "subset", name); out.append(r)
out = pd.concat(out); out.to_csv(os.path.join(a.results, "stats_subsets.csv"), index=False)
t = out[out.method == "res_rf_canon"]
for _, r in t.iterrows():
    star = "*" if r.p_holm < 0.05 else ""
    print(f"{r.subset:12s} {r.scheme[:1]} {int(r.n)} {r.regime:7s} n={int(r.n_circuits):2d} diff={r['diff']:+.4f}{star} p_holm={r.p_holm:.3f}")
