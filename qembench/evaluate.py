"""Evaluation protocols (all leak-free: models never see test circuits or test-circuit labels).

 A  heldout_circuits : train on train circuits of all families, test on unseen circuits
                       (Grover: unseen marked states) -- uniform noise.
 B  leave_family_out : train on two families, test on the third (distribution shift).
 C  device_shift     : train on uniform noise, test on heterogeneous + coherent noise.
 D  fake_device      : train on uniform synthetic noise, test on a calibrated fake IBM device.
"""
import numpy as np
import pandas as pd
from .learned import LEARNED, prepare, train_all, predict_all
from .metrics import hellinger_fidelity, tvd
from .mitigation import BASELINES


def _rows(scheme, held, recs, preps, learned_out):
    rows = []
    for i, (r, d) in enumerate(zip(recs, preps)):
        preds = dict(d["outs"])
        for v, lst in learned_out.items():
            preds[v] = lst[i]
        for m, p in preds.items():
            rows.append(dict(scheme=scheme, held=held, profile=r["profile"], n=r["n"],
                             family=r["family"], cid=r["cid"], scale=r["scale"],
                             regime=r["regime"], rep=r["rep"], method=m,
                             fidelity=hellinger_fidelity(p, r["ideal"]),
                             tvd=tvd(p, r["ideal"]),
                             tprob=float(p[r["target"]]) if r["target"] >= 0 else np.nan))
    return rows


def evaluate_all(records, cfg, seed=0, verbose=True):
    rows = []
    for n in cfg.qubit_sizes:
        R = [r for r in records if r["n"] == n]
        prep_cache = {id(r): prepare(r, cfg.fold_factors) for r in R}
        P = lambda lst: [prep_cache[id(r)] for r in lst]
        tr = [r for r in R if r["profile"] == "uniform" and r["split"] == "train"]
        te_u = [r for r in R if r["profile"] == "uniform" and r["split"] == "test"]
        te_h = [r for r in R if r["profile"] == "hetero" and r["split"] == "test"]
        # A and C share one model set
        models = train_all(tr, P(tr), seed)
        out = predict_all(models, te_u, P(te_u)); rows += _rows("A_heldout_circuits", "", te_u, P(te_u), out)
        if te_h:
            out = predict_all(models, te_h, P(te_h)); rows += _rows("C_device_shift", "", te_h, P(te_h), out)
        te_d = [r for r in R if r["profile"] == "fakedev"]
        if te_d:
            out = predict_all(models, te_d, P(te_d)); rows += _rows("D_fake_device", "", te_d, P(te_d), out)
        # B: leave one family out
        for fam in cfg.families:
            trb = [r for r in tr if r["family"] != fam]
            teb = [r for r in te_u if r["family"] == fam]
            mb = train_all(trb, P(trb), seed)
            out = predict_all(mb, teb, P(teb)); rows += _rows("B_leave_family_out", fam, teb, P(teb), out)
        if verbose:
            print(f"[eval] n={n}: train={len(tr)} test_uniform={len(te_u)} test_hetero={len(te_h)} test_fakedev={len(te_d)}", flush=True)
    return pd.DataFrame(rows)
