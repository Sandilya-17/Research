"""Turn results/raw_results.csv into CSV summaries, LaTeX tables and figures."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .stats import circuit_level, compare
from .mitigation import BASELINES
from .learned import LEARNED

NAMES = {
    "raw": "Raw (unmitigated)", "rem": "REM", "zne_lin13": "ZNE linear (1,3)",
    "zne_lin135": "ZNE linear (1,3,5)", "zne_rich": "ZNE Richardson",
    "rem_zne_lin13": "REM+ZNE linear (1,3)", "rem_zne_lin135": "REM+ZNE linear (1,3,5)",
    "rem_zne_rich": "REM+ZNE Richardson",
    "direct_rf": "Direct RF (original design)", "direct_mlp": "Direct MLP (original design)",
    "res_ridge": "Residual ridge", "res_rf": "Residual RF", "res_mlp": "Residual MLP",
    "res_ridge_canon": "Residual ridge + canon.", "res_rf_canon": "Residual RF + canon.",
    "res_mlp_canon": "Residual MLP + canon.", "mean_canon": "Training-mean (canon.)",
}
SCHEME_NAMES = {"A_heldout_circuits": "A: held-out circuits / marked states",
                "B_leave_family_out": "B: leave-one-family-out",
                "C_device_shift": "C: device shift (heterogeneous + coherent noise)",
                "D_fake_device": "D: calibrated fake IBM device"}
ORDER = BASELINES + LEARNED


def _tex_escape(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def make_all(results_dir, paper_dir, n_boot=10000):
    df = pd.read_csv(results_dir + "/raw_results.csv")
    df["held"] = df["held"].fillna("")
    cl = circuit_level(df)
    tdir, fdir = os.path.join(paper_dir, "tables"), os.path.join(paper_dir, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)

    # 1. pooled paired comparison (circuits pooled over families)
    cl_test = cl[cl.regime != "train"]          # Holm family = all reported non-train comparisons
    pooled = compare(cl_test, ["scheme", "profile", "n", "regime"], n_boot)
    pooled.to_csv(results_dir + "/stats_pooled.csv", index=False)
    perfam = compare(cl_test, ["scheme", "profile", "n", "family", "regime"], n_boot)
    perfam.to_csv(results_dir + "/stats_per_family.csv", index=False)

    # 2. mean fidelity tables (pooled over circuits & families)
    summ = (cl.groupby(["scheme", "n", "regime", "method"])[["fidelity", "tvd", "tprob"]]
              .mean().reset_index())
    summ.to_csv(results_dir + "/summary_means.csv", index=False)
    for scheme, sname in SCHEME_NAMES.items():
        sub = summ[(summ.scheme == scheme) & (summ.regime.isin(["interp", "extrap", "device"]))]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="method", columns=["n", "regime"], values="fidelity")
        piv = piv.reindex([m for m in ORDER if m in piv.index])
        cols = list(piv.columns)
        lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
                 "Method & " + " & ".join(f"{n}q {r}" for n, r in cols) + r" \\", r"\midrule"]
        best = piv.max()
        for m, row in piv.iterrows():
            cells = []
            for c in cols:
                s = f"{row[c]:.3f}"
                cells.append(r"\textbf{" + s + "}" if abs(row[c] - best[c]) < 1e-12 else s)
            sep = r"\midrule" + "\n" if m in ("rem_zne_rich", "mean_canon") else ""
            lines.append(_tex_escape(NAMES[m]) + " & " + " & ".join(cells) + r" \\" + ("\n" + sep if sep else ""))
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(tdir, f"fid_{scheme[0]}.tex"), "w").write("\n".join(lines))

    # 3. paired-statistics table (learned vs best baseline) for scheme A and B/C, interp+extrap
    sel = pooled[pooled.regime.isin(["interp", "extrap", "device"])].copy()
    lines = [r"\begin{longtable}{llrlrrr}", r"\toprule",
             r"Scheme & Method & $n$ & Regime & $\Delta F$ [95\% CI] & $p_{\mathrm{Holm}}$ & Win \\", r"\midrule", r"\endhead"]
    for _, r in sel.sort_values(["scheme", "n", "regime", "method"]).iterrows():
        if r.method in ("direct_mlp", "direct_rf", "mean_canon"):
            continue
        ph = "$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
        lines.append(f"{r.scheme[0]} & {_tex_escape(NAMES[r.method])} & {int(r.n)} & {r.regime} & "
                     f"{r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] & {ph} & {r.win_rate:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    open(os.path.join(tdir, "stats_vs_best.tex"), "w").write("\n".join(lines))

    # 4. Grover success probability (scheme A, interp)
    g = cl[(cl.family == "grover") & (cl.scheme == "A_heldout_circuits") & (cl.regime == "interp")]
    gp = g.groupby(["n", "method"])["tprob"].mean().unstack(0)
    gp = gp.reindex([m for m in ORDER if m in gp.index])
    gp.to_csv(results_dir + "/grover_success_prob.csv")

    # ---- figures ----
    _fig_scale(df, fdir)
    _fig_schemes(pooled, fdir)
    _fig_family(perfam, fdir)
    return pooled, perfam, summ


def _fig_scale(df, fdir):
    d = df[df.scheme == "A_heldout_circuits"]
    d = d.groupby(["n", "scale", "method", "cid"])["fidelity"].mean().reset_index()
    methods = ["raw", "rem", "rem_zne_lin13", "rem_zne_lin135", "res_rf_canon", "res_ridge_canon", "direct_mlp"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, n in zip(axes, sorted(d.n.unique())):
        for m in methods:
            s = d[(d.n == n) & (d.method == m)].groupby("scale")["fidelity"].mean()
            ax.plot(s.index, s.values, marker="o", ms=3, label=NAMES[m])
        ax.axvspan(0.5, 2.0, color="0.92", zorder=0)
        ax.set_title(f"{n} qubits (held-out circuits)"); ax.set_xlabel("noise scale")
    axes[0].set_ylabel("Hellinger fidelity"); axes[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_scale.pdf")); plt.close(fig)


def _fig_schemes(pooled, fdir):
    p = pooled[pooled.regime.isin(["interp", "extrap", "device"])]
    methods = ["res_rf_canon", "res_ridge_canon", "res_rf", "res_mlp_canon", "direct_rf"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, (scheme, sname) in zip(axes, SCHEME_NAMES.items()):
        s = p[p.scheme == scheme]
        labels, vals, errs = [], [], []
        for (n, reg), g in s.groupby(["n", "regime"]):
            for m in methods:
                r = g[g.method == m]
                if len(r):
                    labels.append(f"{m}|{n}{reg}"); vals.append(r["diff"].values[0])
                    errs.append([r["diff"].values[0] - r.ci_lo.values[0], r.ci_hi.values[0] - r["diff"].values[0]])
        # grouped bars
        combos = sorted({(n, reg) for (n, reg), _ in s.groupby(["n", "regime"])})
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            xs, ys, es = [], [], []
            for j, (n, reg) in enumerate(combos):
                r = s[(s.n == n) & (s.regime == reg) & (s.method == m)]
                if len(r):
                    xs.append(j + i * w); ys.append(r["diff"].values[0])
                    es.append([[r["diff"].values[0] - r.ci_lo.values[0]], [r.ci_hi.values[0] - r["diff"].values[0]]])
            if xs:
                ax.bar(xs, ys, w, yerr=np.hstack(es), capsize=1.5, label=NAMES[m] if scheme.startswith("A") else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks([j + 0.4 for j in range(len(combos))])
        ax.set_xticklabels([f"{n}q\n{reg}" for n, reg in combos], fontsize=7)
        ax.set_title(sname.split(":")[0] + ": " + sname.split(":")[1].strip()[:28], fontsize=8)
    axes[0].set_ylabel(r"$\Delta F$ vs best conventional baseline")
    axes[0].legend(fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_schemes.pdf")); plt.close(fig)


def _fig_family(perfam, fdir):
    p = perfam[(perfam.scheme == "A_heldout_circuits") & (perfam.regime == "interp") & (perfam.method == "res_rf_canon")]
    if p.empty:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    labels = [f"{r.family}\n{int(r.n)}q" for _, r in p.iterrows()]
    ax.bar(range(len(p)), p["diff"], yerr=[p["diff"] - p.ci_lo, p.ci_hi - p["diff"]], capsize=2)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(p))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\Delta F$ vs best baseline"); ax.set_title("Residual RF + canon., interpolated noise", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_family.pdf")); plt.close(fig)


SHORT = {"direct_rf": "DRF", "direct_mlp": "DML", "res_ridge": "RG", "res_rf": "RF", "res_mlp": "ML",
         "res_ridge_canon": "RGC", "res_rf_canon": "RFC", "res_mlp_canon": "MLC", "mean_canon": "MEAN"}


def write_numbers(results_dir, paper_dir):
    """Emit paper/numbers.tex so every number quoted in the text is regenerated from results."""
    pooled = pd.read_csv(results_dir + "/stats_pooled.csv")
    lines = ["% AUTO-GENERATED by qembench.report.write_numbers -- do not edit by hand.",
             r"\newcommand{\stat}[5]{\csname S-#1-#2-#3-#4-#5\endcsname}"]

    def d(key, val):
        lines.append(r"\expandafter\def\csname S-" + key + r"\endcsname{" + val + "}")

    def fp(p):
        return r"$<$0.001" if p < 0.001 else f"{p:.3f}"
    for _, r in pooled.iterrows():
        sc, n, rg = r.scheme[0], int(r.n), r.regime
        if r.method not in SHORT:
            continue
        k = f"{sc}-{n}-{rg}-{SHORT[r.method]}"
        d(k + "-F", f"{r.F_method:.3f}"); d(k + "-diff", f"{r['diff']:+.4f}")
        d(k + "-lo", f"{r.ci_lo:+.4f}"); d(k + "-hi", f"{r.ci_hi:+.4f}")
        d(k + "-p", fp(r.p_holm)); d(k + "-win", f"{100*r.win_rate:.0f}")
        d(k + "-ired", f"{100*r.infid_red_vs_raw:.0f}")
        b = f"{sc}-{n}-{rg}-BEST"
        d(b + "-F", f"{r.F_best:.3f}"); d(f"{sc}-{n}-{rg}-RAW-F", f"{r.F_raw:.3f}")
        d(b + "-ired", f"{100*r.infid_red_best_vs_raw:.0f}")
        d(b + "-name", _tex_escape(NAMES[r.best_baseline]))
        d(f"{sc}-{n}-{rg}-N-F", str(int(r.n_circuits)))
    for sc_full in pooled.scheme.unique():
        sc = sc_full[0]
        for m, short in SHORT.items():
            sub = pooled[(pooled.scheme == sc_full) & (pooled.method == m)]
            sub = sub[sub.regime != "train"]
            d(f"{sc}-ALL-ALL-{short}-pos", str(int(((sub["diff"] > 0) & (sub.p_holm < 0.05)).sum())))
            d(f"{sc}-ALL-ALL-{short}-neg", str(int(((sub["diff"] < 0) & (sub.p_holm < 0.05)).sum())))
            d(f"{sc}-ALL-ALL-{short}-tot", str(len(sub)))
    lines.append(r"\newcommand{\ntests}{" + str(len(pooled[pooled.regime != "train"])) + "}")
    g = pd.read_csv(results_dir + "/grover_success_prob.csv").set_index("method")
    for n in (4, 5):
        ideal = np.sin(3 * np.arcsin(2 ** (-n / 2))) ** 2
        d(f"G-{n}-IDEAL-F", f"{ideal:.3f}")
        for m, short in (("raw", "RAW"), ("rem_zne_rich", "ZNE"), ("res_rf_canon", "RFC"), ("res_ridge_canon", "RGC")):
            d(f"G-{n}-{short}-F", f"{g.loc[m, str(n)]:.3f}")
    df = pd.read_csv(results_dir + "/raw_results.csv")
    lines.append(r"\newcommand{\nrows}{" + f"{len(df):,}".replace(",", r"\,") + "}")
    open(os.path.join(paper_dir, "numbers.tex"), "w").write("\n".join(lines) + "\n")
