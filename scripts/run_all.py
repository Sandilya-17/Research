#!/usr/bin/env python3
"""Simulate -> train -> evaluate. Writes results/raw_results.csv and results/config_used.json."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qembench.config import Config, quick_config
from qembench.simulate import simulate_all
from qembench.evaluate import evaluate_all

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
ap.add_argument("--out", default="results")
ap.add_argument("--data", default="data")
ap.add_argument("--force", action="store_true", help="re-simulate even if cached")
a = ap.parse_args()

cfg = quick_config() if a.quick else Config()
if a.quick:
    a.out, a.data = a.out + "_quick", a.data + "_quick"
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
recs = simulate_all(cfg, a.data, force=a.force)
df = evaluate_all(recs, cfg)
df.to_csv(os.path.join(a.out, "raw_results.csv"), index=False)
with open(os.path.join(a.out, "config_used.json"), "w") as f:
    json.dump(cfg.to_dict(), f, indent=2)
print(f"done: {len(df)} rows in {time.time()-t0:.0f}s -> {a.out}/raw_results.csv")
