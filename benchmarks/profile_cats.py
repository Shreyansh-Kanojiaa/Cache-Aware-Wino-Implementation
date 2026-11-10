#!/usr/bin/env python3
"""Where the time goes: median per-category milliseconds of our runner on whole models.

    python3 benchmarks/profile_cats.py MODELS [THREADS] [RUNS] [VARIANT]
    e.g.  python3 benchmarks/profile_cats.py mobilenet_v2,resnet18 1,4 20
"""
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.whole_model import CATEGORIES, PLAN_DIR, export_plan, run_c

models = sys.argv[1].split(",")
threads = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "1,4").split(",")]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
variant = sys.argv[4] if len(sys.argv) > 4 else "v6_freq"
for m in models:
    plan, weights, _, shape = export_plan(m)
    inp = os.path.join(PLAN_DIR, f"{m}.input")
    np.random.RandomState(0).randn(1, *shape).astype(np.float32).tofile(inp)
    for T in threads:
        rows, _ = run_c(plan, weights, inp, variant, None, T, runs, 3)
        med = {c: st.median(float(r[c]) for r in rows) for c in ["total_ms"] + CATEGORIES}
        print(f"{m} T={T} total={med['total_ms']:.1f}ms  " +
              "  ".join(f"{c[:-3]}={med[c]:.1f}" for c in CATEGORIES if med[c] >= 0.05), flush=True)
    os.remove(weights)
