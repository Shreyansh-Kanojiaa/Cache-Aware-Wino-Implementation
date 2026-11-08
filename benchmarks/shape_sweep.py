#!/usr/bin/env python3
"""Time 3x3 conv layer shapes on our engine, ncnn and XNNPACK, to design a model around the
shapes where our engine is fastest.

Probes (random weights, every engine checked against ONNX Runtime):
  conv  C H : 4 x [3x3 stride-1 conv C->C + bias + ReLU] on a C x H x H input
  s2    C H : 3x3 stride-2 conv C->2C + ReLU (H -> H/2)
  pool  C H : 2x2 max pool, then 3x3 stride-1 conv C->2C + ReLU (the other way to halve H)
Writes raw_logs/harness/shape_sweep.csv (one row per probe, engine and thread count).
"""
import argparse
import csv
import os
import statistics as st
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.engine_comparison import check_power, run_ncnn, run_xnnpack, write_ncnn_model
from benchmarks.whole_model import PLAN_DIR, RUNNER, export_plan, in_own_process, run_c, run_ort
from src.harness_runner import ROOT, SRC

CONV = [(96, c) for c in (16, 32, 48, 64)] + [(48, c) for c in (32, 64, 96, 128)] + \
       [(24, c) for c in (64, 128, 192, 256)] + [(12, c) for c in (128, 256, 384, 512)] + \
       [(56, 64), (28, 128), (14, 256), (7, 512), (14, 512), (28, 256)]   # ResNet-style sizes
DOWN = [(96, 32), (96, 64), (48, 64), (48, 128), (24, 128), (24, 256)]


def probe_onnx(kind, C, H, path):
    rng = np.random.RandomState(0)
    nodes, inits = [], []
    def conv(x, cin, cout, s, name):
        w = (rng.randn(cout, cin, 3, 3) * np.sqrt(2.0 / (9 * cin))).astype(np.float32)
        b = (rng.randn(cout) * 0.1).astype(np.float32)
        inits.extend([numpy_helper.from_array(w, name + "_w"), numpy_helper.from_array(b, name + "_b")])
        nodes.append(helper.make_node("Conv", [x, name + "_w", name + "_b"], [name], kernel_shape=[3, 3],
                                      strides=[s, s], pads=[1, 1, 1, 1], dilations=[1, 1], group=1))
        nodes.append(helper.make_node("Relu", [name], [name + "_r"]))
        return name + "_r"
    x = "input"
    if kind == "conv":
        for i in range(4):
            x = conv(x, C, C, 1, f"c{i}")
        cout, ho = C, H
    elif kind == "s2":
        x, cout, ho = conv(x, C, 2 * C, 2, "c0"), 2 * C, H // 2
    else:
        nodes.append(helper.make_node("MaxPool", [x], ["mp"], kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0], ceil_mode=0))
        x, cout, ho = conv("mp", C, 2 * C, 1, "c0"), 2 * C, H // 2
    g = helper.make_graph(nodes, "probe", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, H, H])],
                          [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, cout, ho, ho])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--threads", default="4,1")
    p.add_argument("--out", default="raw_logs/harness/shape_sweep.csv")
    p.add_argument("--only", default="", help="comma list of probe names to run, e.g. probe_conv_c128_h12")
    args = p.parse_args()
    os.system(f"gcc -O3 -march=native -fopenmp -o {RUNNER} {os.path.join(os.path.dirname(SRC), 'model_runner.c')} -lm")
    check_power()
    probes = [("conv", C, H) for H, C in CONV] + [(k, C, H) for H, C in DOWN for k in ("s2", "pool")]
    if args.only:
        probes = [x for x in probes if f"probe_{x[0]}_c{x[1]}_h{x[2]}" in args.only.split(",")]
    fields = ["kind", "C", "H", "engine", "threads", "n", "median_ms", "std_ms", "err"]
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for kind, C, H in probes:
            name = f"probe_{kind}_c{C}_h{H}"
            probe_onnx(kind, C, H, os.path.join(ROOT, name + ".onnx"))
            plan, weights, onnx_path, shape = export_plan(name)
            inp = os.path.join(PLAN_DIR, name + ".input")
            np.random.RandomState(1).randn(1, *shape).astype(np.float32).tofile(inp)
            param, nbin = os.path.join(PLAN_DIR, name + ".param"), os.path.join(PLAN_DIR, name + ".ncnn.bin")
            in_blob, out_blobs = write_ncnn_model(plan, weights, param, nbin)
            for T in (int(t) for t in args.threads.split(",")):
                lats, ref = in_own_process(run_ort, onnx_path, inp, shape, T, args.runs, args.warmup)
                res = [("onnxruntime", lats, ref)]
                for eng, variant in (("ours_wino", "v6_freq"), ("ours_direct", "direct_packed")):
                    rows, out = run_c(plan, weights, inp, variant, None, T, args.runs, args.warmup)
                    res.append((eng, [float(r["total_ms"]) for r in rows], out))
                rows, out = run_xnnpack(plan, weights, inp, T, args.runs, args.warmup)
                res.append(("xnnpack", [float(r["total_ms"]) for r in rows], out))
                for eng, fp16 in (("ncnn", True), ("ncnn_fp32", False)):
                    lats, out = in_own_process(run_ncnn, param, nbin, in_blob, out_blobs, inp, shape, T,
                                               args.runs, args.warmup, True, fp16)
                    res.append((eng, lats, out))
                check_power()
                for eng, lats, out in res:
                    err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
                    w.writerow({"kind": kind, "C": C, "H": H, "engine": eng, "threads": T, "n": len(lats),
                                "median_ms": round(st.median(lats), 3), "std_ms": round(st.stdev(lats), 3),
                                "err": f"{err:.1e}"})
                f.flush()
                print(name, f"T={T}", " ".join(f"{e}={st.median(l):.2f}" for e, l, _ in res), flush=True)
            for path in (weights, nbin, onnx_path):
                os.remove(path)
    print("done")


if __name__ == "__main__":
    main()
