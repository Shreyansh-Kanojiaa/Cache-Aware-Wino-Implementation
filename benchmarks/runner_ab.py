"""Interleaved timing of two model_runner builds (base = the previous commit built to build/model_runner_base), v6_freq, T=1 and T=4.
Usage: time_ab.py MODELS OUT_CSV; configs below."""
import os, subprocess, sys, csv
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, "/home/j/wino-own")
from benchmarks import whole_model as wm
BASE = "/home/j/wino-own/build/model_runner_base"
CONFIGS = [("base", BASE, []), ("new", wm.RUNNER, [])]
models, out_csv = sys.argv[1].split(","), sys.argv[2]
BLOCKS, RUNS = 4, 15
rows = []
for m in models:
    plan, weights, _, _ = wm.export_plan(m)
    inp = os.path.join(wm.PLAN_DIR, f"{m}.input")
    for T in (1, 4):
        t = {c[0]: [] for c in CONFIGS}
        for blk in range(BLOCKS):
            order = CONFIGS if blk % 2 == 0 else CONFIGS[::-1]
            for name, binary, extra in order:
                subprocess.run([binary, "--plan", plan, "--weights", weights, "--input", inp, "--output", "/tmp/ta.bin",
                                "--csv", "/tmp/ta.csv", "--variant", "v6_freq", "--T", str(T), "--warmup", "3",
                                "--runs", str(RUNS)] + extra, stderr=subprocess.DEVNULL, check=True)
                for r in csv.DictReader(open("/tmp/ta.csv")):
                    t[name].append(float(r["total_ms"]))
                    rows.append(dict(model=m, T=T, config=name, block=blk, **r))
        b = np.median(t["base"])
        msg = f"{m} T={T} base={b:.1f}"
        for name in t:
            if name != "base":
                v = np.median(t[name])
                msg += f" {name}={v:.1f} ({100 * (v - b) / b:+.1f}%, p={mannwhitneyu(t['base'], t[name]).pvalue:.1g})"
        print(msg, flush=True)
    os.remove(weights)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("done", flush=True)
