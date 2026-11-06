#!/usr/bin/env python3
"""Time the fused kernel at each m for every distinct stride-1 3x3 layer in the
E2E models. This is the data behind the tile-selection policy.

Writes raw_logs/harness/layer_m_grid.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.e2e_benchmark import MODELS, _get_conv3x3_layers
from src.harness_runner import conv_output_size, measure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", default="4,1")
    p.add_argument("--raw", default="raw_logs/harness/layer_m_grid.csv")
    args = p.parse_args()

    shapes = sorted({(l["c_in"], l["c_out"], conv_output_size(l["h_in"], 1, l["padding"]),
                      conv_output_size(l["w_in"], 1, l["padding"]))
                     for model in MODELS for l in _get_conv3x3_layers(model)
                     if l["stride"] == 1})
    print(f"{len(shapes)} distinct shapes", flush=True)
    for T in (int(t) for t in args.threads.split(",")):
        for c_in, c_out, h, w in shapes:
            means = []
            for m in (2, 4, 6):
                lats = measure("fused", m, T, h, w, c_in, c_out, args.runs, args.warmup,
                               cell_id=f"grid_m{m}_T{T}_c{c_in}x{c_out}_{h}x{w}", raw_path=args.raw)
                means.append(sum(lats) / len(lats))
            print(f"T={T} {c_in}x{c_out}@{h}x{w}: " + " ".join(f"m{m}={x:.1f}" for m, x in zip((2, 4, 6), means)), flush=True)


if __name__ == "__main__":
    main()
