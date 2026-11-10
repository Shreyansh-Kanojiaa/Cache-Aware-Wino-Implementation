"""Interleaved A/B timing of two model_runner builds, plus a correctness check of the new one.

    python3 benchmarks/runner_ab.py MODELS OUT_CSV [THREADS] [BLOCKS] [RUNS]
    e.g.  python3 benchmarks/runner_ab.py mobilenet_v2,resnet18 raw_logs/harness/ab_x.csv 1,4

base = build/model_runner_base (keep it built from the commit you started from), new = build/model_runner.
Blocks alternate base/new order so drift hits both; p is a Mann-Whitney test on all runs.
err is the new build's largest output difference from ONNX Runtime, relative to the largest output.
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks import whole_model as wm

BASE = os.path.join(os.path.dirname(wm.RUNNER), "model_runner_base")
CONFIGS = [("base", BASE, []), ("new", wm.RUNNER, [])]


def main():
    models, out_csv = sys.argv[1].split(","), sys.argv[2]
    threads = [int(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "1,4").split(",")]
    BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    RUNS = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        out_bin, run_csv = os.path.join(tmp, "out.bin"), os.path.join(tmp, "runs.csv")
        for m in models:
            plan, weights, onnx_path, shape = wm.export_plan(m)
            inp = os.path.join(wm.PLAN_DIR, f"{m}.input")
            np.random.RandomState(0).randn(1, *shape).astype(np.float32).tofile(inp)
            _, ref = wm.in_own_process(wm.run_ort, onnx_path, inp, shape, 4, 1, 0)
            for T in threads:
                t = {c[0]: [] for c in CONFIGS}
                err = 0.0
                for blk in range(BLOCKS):
                    order = CONFIGS if blk % 2 == 0 else CONFIGS[::-1]
                    for name, binary, extra in order:
                        subprocess.run([binary, "--plan", plan, "--weights", weights, "--input", inp, "--output", out_bin,
                                        "--csv", run_csv, "--variant", "v6_freq", "--T", str(T), "--warmup", "3",
                                        "--runs", str(RUNS)] + extra, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                        for r in csv.DictReader(open(run_csv)):
                            t[name].append(float(r["total_ms"]))
                            rows.append(dict(model=m, T=T, config=name, block=blk, **r))
                        if name == "new":
                            out = np.fromfile(out_bin, dtype=np.float32)
                            err = max(err, float(np.max(np.abs(out - ref)) / np.max(np.abs(ref))))
                b, v = np.median(t["base"]), np.median(t["new"])
                print(f"{m} T={T} base={b:.2f} new={v:.2f} ({100 * (v - b) / b:+.1f}%, "
                      f"p={mannwhitneyu(t['base'], t['new']).pvalue:.1g}) err={err:.1e}", flush=True)
            os.remove(weights)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
