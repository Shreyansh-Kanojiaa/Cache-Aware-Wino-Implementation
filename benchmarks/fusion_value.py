#!/usr/bin/env python3
"""Measure what fusion itself is worth, with everything else held equal.

All three modes are v5_batched with the same packed inner loop and cache blocking.
Only the batch size changes:
  fused      batch 1: each tile is finished before the next starts
  nonfused   batch of all tiles: all input transforms, then multiply, then output
  batched    the automatic batch size

Threads differ by design at T>1: fused gives each thread its own tiles, while
nonfused shares one batch and splits the weight blocks between threads.

Writes raw_logs/harness/fusion_value.csv, summary/harness/fusion_value.csv and
summary/harness/fusion_value_models.csv. Model totals cover stride-1 3x3 layers only.
"""
import argparse
import collections
import csv
import os
import statistics as st
import sys

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.e2e_benchmark import MODELS, _get_conv3x3_layers
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.harness_runner import conv_output_size, measure

ALL_TILES = 1000000  # larger than any tile count, so the kernel uses one shared batch
MODES = {"nonfused": ALL_TILES, "fused": 1, "batched": None}
CONFIGS = [(16, 32), (32, 16), (32, 32), (32, 64), (64, 32), (64, 64), (128, 128)]
SIZES = [14, 28, 56, 112, 224]


def working_memory_bytes(c_in, c_out, h, m, threads, batch):
    """Bytes held for transformed inputs and accumulators across all threads."""
    tiles = ((h + m - 1) // m) ** 2
    per_tile = (c_in + ((c_out + 7) // 8) * 8) * (m + 2) ** 2 * 4
    return (tiles if batch == ALL_TILES else threads) * per_tile


def gain(baseline, other):
    return round((st.median(baseline) - st.median(other)) / st.median(baseline) * 100.0, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--threads", default="4,1")
    p.add_argument("--raw", default="raw_logs/harness/fusion_value.csv")
    args = p.parse_args()

    tiler = CacheAdaptiveAutotiler()
    layers_by_model = {model: [l for l in _get_conv3x3_layers(model) if l["stride"] == 1]
                       for model in MODELS}
    shapes = collections.OrderedDict()
    for layers in layers_by_model.values():
        for l in layers:
            h = conv_output_size(l["h_in"], 1, l["padding"])
            td = tiler.select_best_tile(l["c_in"], l["c_out"])["selected_tile"]["tile"]
            shapes[(l["c_in"], l["c_out"], h, td - 2)] = "model_layer"
    for (c_in, c_out) in CONFIGS:
        for h in SIZES:
            shapes.setdefault((c_in, c_out, h, 4), "taskM_config")

    rows, medians = [], {}
    threads = [int(t) for t in args.threads.split(",")]
    for T in threads:
        for (c_in, c_out, h, m), group in shapes.items():
            lats = {}
            for mode, batch in MODES.items():
                lats[mode] = measure("v5_batched", m, T, h, h, c_in, c_out, args.runs, args.warmup,
                                     cell_id=f"fv_{mode}_m{m}_T{T}_c{c_in}x{c_out}_{h}",
                                     raw_path=args.raw, batch=batch)
                medians[(T, c_in, c_out, h, mode)] = st.median(lats[mode])
            rows.append({
                "group": group, "c_in": c_in, "c_out": c_out, "h": h, "m": m, "threads": T,
                "n": args.runs,
                "nonfused_median_ms": round(st.median(lats["nonfused"]), 3),
                "fused_median_ms": round(st.median(lats["fused"]), 3),
                "batched_median_ms": round(st.median(lats["batched"]), 3),
                "fused_gain_pct": gain(lats["nonfused"], lats["fused"]),
                "fused_p": f'{stats.ttest_ind(lats["fused"], lats["nonfused"], equal_var=False)[1]:.2e}',
                "batched_gain_pct": gain(lats["nonfused"], lats["batched"]),
                "batched_p": f'{stats.ttest_ind(lats["batched"], lats["nonfused"], equal_var=False)[1]:.2e}',
                "nonfused_work_mem_kb": working_memory_bytes(c_in, c_out, h, m, T, ALL_TILES) // 1024,
                "fused_work_mem_kb": working_memory_bytes(c_in, c_out, h, m, T, 1) // 1024,
            })
            r = rows[-1]
            print(f"T={T} {c_in}x{c_out}@{h} m{m}: nonfused={r['nonfused_median_ms']} "
                  f"fused={r['fused_median_ms']} batched={r['batched_median_ms']} "
                  f"fused_gain={r['fused_gain_pct']}% batched_gain={r['batched_gain_pct']}%", flush=True)

    os.makedirs("summary/harness", exist_ok=True)
    with open("summary/harness/fusion_value.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    model_rows = []
    for T in threads:
        for model, layers in layers_by_model.items():
            total = {mode: sum(medians[(T, l["c_in"], l["c_out"],
                                        conv_output_size(l["h_in"], 1, l["padding"]), mode)]
                               for l in layers) for mode in MODES}
            model_rows.append({"model": model, "threads": T,
                               **{f"{k}_ms": round(v, 1) for k, v in total.items()},
                               "fused_gain_pct": round((total["nonfused"] - total["fused"]) / total["nonfused"] * 100, 2),
                               "batched_gain_pct": round((total["nonfused"] - total["batched"]) / total["nonfused"] * 100, 2)})
            print(model_rows[-1], flush=True)
    with open("summary/harness/fusion_value_models.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_rows)


if __name__ == "__main__":
    main()
