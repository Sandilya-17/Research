#!/usr/bin/env python3
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qembench.report import make_all, write_numbers
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--paper", default="paper")
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.results, "config_used.json")))
make_all(a.results, a.paper, n_boot=cfg.get("n_boot", 10000))
write_numbers(a.results, a.paper)
print("report written")
