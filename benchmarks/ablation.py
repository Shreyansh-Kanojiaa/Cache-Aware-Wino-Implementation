#!/usr/bin/env python3
"""Ablation: what each optimization in the runner is worth, by switching one off at a time.

    python3 benchmarks/ablation.py MODELS OUT_CSV [THREADS] [BLOCKS] [RUNS]
    e.g.  python3 benchmarks/ablation.py resnet18,mobilenet_v2 raw_logs/harness/ablation.csv 1,4 4 10

Configs (all the current build unless noted), timed in interleaved blocks whose order rotates:
  full          everything on (the reference)
  tile_f43      every 3x3 stride-1 conv uses F(4,3) (no per-layer tile choice)
  tile_f63      every 3x3 stride-1 conv uses F(6,3)
  tile_f23      every 3x3 stride-1 conv uses F(2,3)
  no_winograd   3x3 stride-1 convs use the direct kernel
  no_pad_link   Winograd convs don't write into the next conv's padded input
  no_act_fuse   ReLU / Clip / SiLU run as their own pass
  no_pwdw_fuse  1x1 expansion and depthwise run as two ops
  no_recycle    every tensor gets its own buffer
  previous      the engine before the depthwise / 1x1 / single-core work (build/model_runner_base)
Prints median ms per config and the change against full (Mann-Whitney p over all runs).
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
# name, binary, variant, extra args, forced tile size (None: the runner chooses)
CONFIGS = [
    ("full", wm.RUNNER, "v6_freq", [], None),
    ("tile_f43", wm.RUNNER, "v6_freq", ["--plan-m", "1"], 4),
    ("tile_f63", wm.RUNNER, "v6_freq", ["--plan-m", "1"], 6),
    ("tile_f23", wm.RUNNER, "v6_freq", ["--plan-m", "1"], 2),
    ("no_winograd", wm.RUNNER, "direct_packed", [], None),
    ("no_pad_link", wm.RUNNER, "v6_freq", ["--pad-link", "0"], None),
    ("no_act_fuse", wm.RUNNER, "v6_freq", ["--fuse-act", "0"], None),
    ("no_pwdw_fuse", wm.RUNNER, "v6_freq", ["--fuse-pwdw", "0"], None),
    ("no_recycle", wm.RUNNER, "v6_freq", ["--recycle", "0"], None),
    ("previous", BASE, "v6_freq", [], None),
]


def plan_with_m(plan, m, out):
    """A copy of the plan whose 3x3 stride-1 convs all ask for tile size m."""
    with open(plan) as f, open(out, "w") as g:
        for line in f:
            p = line.split()
            if p and p[0] == "CONV" and p[5] == "3" and p[6] == "1":
                p[8] = str(m)   # CONV out in cin cout k s p m woff boff
                line = " ".join(p) + "\n"
            g.write(line)
    return out


def main():
    models, out_csv = sys.argv[1].split(","), sys.argv[2]
    threads = [int(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "1,4").split(",")]
    blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        out_bin, run_csv = os.path.join(tmp, "out.bin"), os.path.join(tmp, "runs.csv")
        for m in models:
            plan, weights, onnx_path, shape = wm.export_plan(m)
            inp = os.path.join(wm.PLAN_DIR, f"{m}.input")
            np.random.RandomState(0).randn(1, *shape).astype(np.float32).tofile(inp)
            _, ref = wm.in_own_process(wm.run_ort, onnx_path, inp, shape, 4, 1, 0)
            plans = {None: plan}
            for tile in (2, 4, 6):
                plans[tile] = plan_with_m(plan, tile, os.path.join(tmp, f"m{tile}.plan"))
            for T in threads:
                t = {c[0]: [] for c in CONFIGS}
                err = {c[0]: 0.0 for c in CONFIGS}
                for blk in range(blocks):
                    k = blk % len(CONFIGS)
                    for name, binary, variant, extra, tile in CONFIGS[k:] + CONFIGS[:k]:
                        subprocess.run([binary, "--plan", plans[tile], "--weights", weights, "--input", inp,
                                        "--output", out_bin, "--csv", run_csv, "--variant", variant, "--T", str(T),
                                        "--warmup", "3", "--runs", str(runs)] + extra,
                                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True)
                        for r in csv.DictReader(open(run_csv)):
                            t[name].append(float(r["total_ms"]))
                            rows.append(dict(model=m, T=T, config=name, block=blk, **r))
                        out = np.fromfile(out_bin, dtype=np.float32)
                        err[name] = max(err[name], float(np.max(np.abs(out - ref)) / np.max(np.abs(ref))))
                full = np.median(t["full"])
                for name in t:
                    v = np.median(t[name])
                    p = "" if name == "full" else f", p={mannwhitneyu(t['full'], t[name]).pvalue:.1g}"
                    print(f"{m} T={T} {name:13s} {v:9.2f} ms ({100 * (v - full) / full:+6.1f}%{p}) err={err[name]:.1e}", flush=True)
            os.remove(weights)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
