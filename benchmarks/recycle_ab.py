"""Interleaved A/B timing: --recycle 0 vs 1, v6_freq, all models, T=1 and T=4."""
import os, subprocess, sys, csv, statistics
import numpy as np
from scipy.stats import mannwhitneyu
sys.path.insert(0, "/home/j/wino-own/benchmarks")
import whole_model as wm
models = sys.argv[1].split(",")
out_csv = sys.argv[2]
BLOCKS, RUNS = 4, 15
rows = []
for m in models:
    plan, weights, _, _ = wm.export_plan(m)
    inp = os.path.join(wm.PLAN_DIR, f"{m}.input")
    for T in (1, 4):
        t = {0: [], 1: []}
        for blk in range(BLOCKS):
            for rc in ((0, 1) if blk % 2 == 0 else (1, 0)):
                subprocess.run([wm.RUNNER, "--plan", plan, "--weights", weights, "--input", inp, "--output", "/tmp/tr.bin",
                                "--csv", "/tmp/tr.csv", "--variant", "v6_freq", "--T", str(T), "--warmup", "3",
                                "--runs", str(RUNS), "--recycle", str(rc)], stderr=subprocess.DEVNULL, check=True)
                for r in csv.DictReader(open("/tmp/tr.csv")):
                    t[rc].append(float(r["total_ms"]))
                    rows.append(dict(model=m, T=T, recycle=rc, block=blk, **r))
        a, b = np.median(t[0]), np.median(t[1])
        p = mannwhitneyu(t[0], t[1]).pvalue
        print(f"{m} T={T} off={a:.1f} on={b:.1f} change={100*(b-a)/a:+.1f}% p={p:.2g}", flush=True)
    os.remove(weights)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("done", flush=True)
