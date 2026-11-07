#!/usr/bin/env python3
"""Time single 3x3 conv layers in ONNX Runtime and the C kernels, side by side.

This is the like-for-like comparison: the same layer shape and thread count, with
no other model ops involved. ONNX Runtime gets its own process each time, so its
worker threads can't overlap the C kernels' timings.

Writes raw_logs/harness/single_layer_vs_ort.csv and
summary/harness/single_layer_vs_ort.csv.
"""
import argparse
import csv
import multiprocessing as mp
import os
import statistics as st
import sys
import time

import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.harness_runner import conv_output_size, measure

# c_in, c_out, input size, stride
LAYERS = [
    (64, 128, 56, 2), (128, 256, 28, 2), (256, 512, 14, 2),
    (32, 32, 56, 1), (64, 64, 56, 1), (128, 128, 28, 1),
    (256, 256, 14, 1), (512, 512, 14, 1), (512, 512, 7, 1),
]


def ort_conv(c_in, c_out, size, stride, threads, runs, warmup):
    np.random.seed(0)
    weights = onh.from_array(np.random.rand(c_out, c_in, 3, 3).astype(np.float32), "w")
    node = oh.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                        strides=[stride, stride])
    out = conv_output_size(size, stride)
    graph = oh.make_graph([node], "conv",
                          [oh.make_tensor_value_info("x", 1, [1, c_in, size, size])],
                          [oh.make_tensor_value_info("y", 1, [1, c_out, out, out])], [weights])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)], ir_version=8)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    x = np.random.rand(1, c_in, size, size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"x": x})
    lats = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {"x": x})
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--threads", default="4,1")
    p.add_argument("--raw", default="raw_logs/harness/single_layer_vs_ort.csv")
    p.add_argument("--summary", default="summary/harness/single_layer_vs_ort.csv")
    args = p.parse_args()

    rows = []
    ctx = mp.get_context("spawn")
    for T in (int(t) for t in args.threads.split(",")):
        for c_in, c_out, size, stride in LAYERS:
            out = conv_output_size(size, stride)
            cell = f"{c_in}x{c_out}@{size}s{stride}"
            with ctx.Pool(1) as pool:
                ort_lats = pool.apply(ort_conv, (c_in, c_out, size, stride, T, args.runs, args.warmup))
            times = {"onnxruntime": ort_lats}
            times["direct_packed"] = measure("direct_packed", 4, T, out, out, c_in, c_out, args.runs,
                                             args.warmup, cell_id=f"slo_dp_T{T}_{cell}",
                                             raw_path=args.raw, stride=stride)
            times["neon_direct"] = measure("neon_direct", 4, T, out, out, c_in, c_out, args.runs,
                                           args.warmup, cell_id=f"slo_nd_T{T}_{cell}", raw_path=args.raw)
            if stride == 1:
                times["v5_batched_m4"] = measure("v5_batched", 4, T, out, out, c_in, c_out, args.runs,
                                                 args.warmup, cell_id=f"slo_v5_T{T}_{cell}",
                                                 raw_path=args.raw)
            for kernel, lats in times.items():
                rows.append({"layer": cell, "stride": stride, "threads": T, "kernel": kernel,
                             "median_ms": round(st.median(lats), 3),
                             "mean_ms": round(st.mean(lats), 3),
                             "std_ms": round(st.stdev(lats), 3), "n": len(lats)})
            print(f"T={T} {cell}: " + " ".join(f"{k}={st.median(v):.2f}" for k, v in times.items()),
                  flush=True)

    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
