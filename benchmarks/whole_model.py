#!/usr/bin/env python3
"""Compare whole models: ONNX Runtime against our C engine (src/model_runner.c).

Both engines run the same ONNX graph, weights and input, and each engine's final
output is compared with ONNX Runtime's. ONNX Runtime gets its own process.

Writes raw_logs/harness/whole_model.csv and summary/harness/whole_model.csv.
Generated plans and weights go in build/models/, which git ignores.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import datetime
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnx
import onnx.numpy_helper as onh
import onnxruntime as ort
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.e2e_benchmark import WHOLE_MODELS as MODELS, _export_onnx
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.harness_runner import ROOT, SRC, build

RUNNER = os.path.join(ROOT, "build", "model_runner")
PLAN_DIR = os.path.join(ROOT, "build", "models")
ALL_TILES = 1000000
# name, variant, batch, extra model_runner arguments
ENGINES = [
    ("c_v6_freq", "v6_freq", None, []),
    ("c_v6_fc16", "v6_freq", None, ["--fc-fp16", "1"]),  # 16-bit fully connected weights, like ncnn's default
    ("c_v5_batched", "v5_batched", None, []),
    ("c_strict_fusion", "v5_batched", 1, []),
    ("c_nonfused", "v5_batched", ALL_TILES, []),
    ("c_direct_packed", "direct_packed", None, []),
]
CATEGORIES = ["conv3x3_s1_ms", "conv_other_ms", "pad_ms", "relu_ms", "pool_ms", "add_gap_ms", "gemm_ms",
              "conv_depthwise_ms", "clip_silu_ms", "concat_upsample_ms"]


def in_own_process(fn, *args):
    """Run fn in a fresh process. Unlike multiprocessing.Pool, a crash raises instead of hanging."""
    if sys.version_info < (3, 7):   # no mp_context yet (the Jetson's Python 3.6)
        with mp.get_context("spawn").Pool(1) as pool:
            return pool.apply(fn, args)
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result()


def export_plan(model_name):
    """Translate the ONNX graph into the plan and weights files model_runner reads."""
    os.makedirs(PLAN_DIR, exist_ok=True)
    onnx_path = os.path.join(ROOT, f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        _export_onnx(model_name, onnx_path)
    graph = onnx.load(onnx_path).graph
    inits = {i.name: onh.to_array(i).astype(np.float32) for i in graph.initializer}
    # Older exporters (PyTorch 1.10 on the Jetson) write Clip bounds as Constant nodes
    inits.update({n.output[0]: onh.to_array(n.attribute[0].t).astype(np.float32)
                  for n in graph.node if n.op_type == "Constant"})
    consumers = collections.Counter(x for n in graph.node for x in n.input)
    tiler = CacheAdaptiveAutotiler()

    ids, shapes, lines = {}, {}, []
    next_id = [0]
    offset = [0]
    weights_path = os.path.join(PLAN_DIR, f"{model_name}.bin")
    plan_path = os.path.join(PLAN_DIR, f"{model_name}.plan")

    def new_tensor(name, shape):
        ids[name] = next_id[0]
        shapes[name] = shape
        lines.append(f"TENSOR {next_id[0]} {shape[0]} {shape[1]} {shape[2]}")
        next_id[0] += 1
        return ids[name]

    aliased = set()  # Split halves share their input's memory, so nothing writes them in place
    sigmoids = {}

    def reuse(name, source, shape):
        ids[name] = ids[source]
        shapes[name] = shape
        return ids[name]

    def attrs(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    graph_input = graph.input[0]
    _, c, h, w = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]
    lines.append(f"INPUT {new_tensor(graph_input.name, (c, h, w))}")

    with open(weights_path, "wb") as wf:
        def store(array):
            start = offset[0]
            array.tofile(wf)
            offset[0] += array.size
            return start

        for node in graph.node:
            if node.op_type == "Constant":
                continue
            op, a = node.op_type, attrs(node)
            name = node.output[0]
            src = node.input[0]
            if op == "Conv" and a["group"] != 1:
                k, s, p = a["kernel_shape"], a["strides"], a["pads"]
                assert k[0] == k[1] and s[0] == s[1] and len(set(p)) == 1 and set(a["dilations"]) == {1}
                W = inits[node.input[1]]
                ci, hi, wi = shapes[src]
                assert a["group"] == ci and W.shape[:2] == (ci, 1)  # depthwise only
                ho = (hi + 2 * p[0] - k[0]) // s[0] + 1
                wo = (wi + 2 * p[0] - k[0]) // s[0] + 1
                woff = store(W)
                boff = store(inits[node.input[2]]) if len(node.input) > 2 and node.input[2] else -1
                out = new_tensor(name, (ci, ho, wo))
                lines.append(f"DWCONV {out} {ids[src]} {ci} {k[0]} {s[0]} {p[0]} {woff} {boff}")
            elif op == "Conv":
                k, s, p = a["kernel_shape"], a["strides"], a["pads"]
                assert a["group"] == 1 and set(a["dilations"]) == {1}
                assert k[0] == k[1] and s[0] == s[1] and len(set(p)) == 1
                W = inits[node.input[1]]
                cout, cin = W.shape[:2]
                _, hi, wi = shapes[src]
                ho = (hi + 2 * p[0] - k[0]) // s[0] + 1
                wo = (wi + 2 * p[0] - k[0]) // s[0] + 1
                m = 0
                if k[0] == 3 and s[0] == 1:
                    assert p[0] == 1
                    m = tiler.select_best_tile(cin, cout)["selected_tile"]["tile"] - 2
                woff = store(W)
                boff = store(inits[node.input[2]]) if len(node.input) > 2 and node.input[2] else -1
                out = new_tensor(name, (cout, ho, wo))
                lines.append(f"CONV {out} {ids[src]} {cin} {cout} {k[0]} {s[0]} {p[0]} {m} {woff} {boff}")
            elif op == "Relu":
                in_place = consumers[src] == 1 and src != graph_input.name and src not in aliased
                out = reuse(name, src, shapes[src]) if in_place else new_tensor(name, shapes[src])
                lines.append(f"RELU {out} {ids[src]}")
            elif op == "Clip":
                lo, hi = (float(inits[x]) for x in node.input[1:3])
                in_place = consumers[src] == 1 and src != graph_input.name and src not in aliased
                out = reuse(name, src, shapes[src]) if in_place else new_tensor(name, shapes[src])
                lines.append(f"CLIP {out} {ids[src]} {lo} {hi}")
            elif op == "Sigmoid":
                assert consumers[name] == 1
                sigmoids[name] = src  # emitted as SILU at the Mul that follows
            elif op == "Mul":
                x, y = node.input[0], node.input[1]
                sig, val = (y, x) if y in sigmoids else (x, y)
                assert sigmoids[sig] == val  # only x * sigmoid(x)
                in_place = consumers[val] == 2 and val != graph_input.name and val not in aliased
                out = reuse(name, val, shapes[val]) if in_place else new_tensor(name, shapes[val])
                lines.append(f"SILU {out} {ids[val]}")
            elif op == "Split":
                assert a["axis"] == 1 and a["num_outputs"] == 2 and consumers[src] == 1
                ci, hi, wi = shapes[src]
                assert ci % 2 == 0
                x, y = new_tensor(node.output[0], (ci // 2, hi, wi)), new_tensor(node.output[1], (ci // 2, hi, wi))
                aliased.update(node.output)
                lines.append(f"SPLIT {x} {y} {ids[src]}")
            elif op == "Concat":
                assert a["axis"] == 1
                hi, wi = shapes[src][1:]
                assert all(shapes[x][1:] == (hi, wi) for x in node.input)
                out = new_tensor(name, (sum(shapes[x][0] for x in node.input), hi, wi))
                lines.append(f"CONCAT {out} {len(node.input)} " + " ".join(str(ids[x]) for x in node.input))
            elif op == "Resize":
                assert a["mode"] == b"nearest" and a["coordinate_transformation_mode"] == b"asymmetric"
                assert inits[node.input[2]].tolist() == [1.0, 1.0, 2.0, 2.0]
                ci, hi, wi = shapes[src]
                out = new_tensor(name, (ci, hi * 2, wi * 2))
                lines.append(f"UPSAMPLE {out} {ids[src]}")
            elif op == "MaxPool":
                k, s, p = a["kernel_shape"][0], a["strides"][0], a["pads"][0]
                assert len(set(a["pads"])) == 1 and a["ceil_mode"] == 0
                ci, hi, wi = shapes[src]
                out = new_tensor(name, (ci, (hi + 2 * p - k) // s + 1, (wi + 2 * p - k) // s + 1))
                lines.append(f"MAXPOOL {out} {ids[src]} {k} {s} {p}")
            elif op == "AveragePool":
                assert a["kernel_shape"] == [1, 1] and a["strides"] == [1, 1]  # identity
                reuse(name, src, shapes[src])
            elif op == "Add":
                b = node.input[1]
                if consumers[src] == 1 and src not in aliased:
                    out = reuse(name, src, shapes[src])
                elif consumers[b] == 1 and b not in aliased:
                    out = reuse(name, b, shapes[b])
                else:
                    out = new_tensor(name, shapes[src])
                lines.append(f"ADD {out} {ids[src]} {ids[b]}")
            elif op in ("ReduceMean", "GlobalAveragePool"):
                if op == "ReduceMean":
                    assert {int(x) % 4 for x in inits[node.input[1]].tolist()} == {2, 3}
                ci = shapes[src][0]
                out = new_tensor(name, (ci, 1, 1))
                lines.append(f"GAP {out} {ids[src]}")
            elif op in ("Reshape", "Flatten"):
                ci, hi, wi = shapes[src]
                reuse(name, src, (ci * hi * wi, 1, 1))
            elif op == "Gemm":
                assert a["transB"] == 1 and a.get("alpha", 1.0) == 1.0 and a.get("beta", 1.0) == 1.0
                W = inits[node.input[1]]
                nout, nin = W.shape
                assert shapes[src][0] == nin
                woff = store(W)
                boff = store(inits[node.input[2]]) if len(node.input) > 2 and node.input[2] else -1
                out = new_tensor(name, (nout, 1, 1))
                lines.append(f"GEMM {out} {ids[src]} {nin} {nout} {woff} {boff}")
            else:
                raise ValueError(f"Unsupported op {op}")

    lines.extend(f"OUTPUT {ids[o.name]}" for o in graph.output)
    with open(plan_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return plan_path, weights_path, onnx_path, (c, h, w)


def run_ort(onnx_path, input_path, shape, threads, runs, warmup):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
    x = np.fromfile(input_path, dtype=np.float32).reshape((1,) + tuple(shape))
    name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {name: x})
    lats = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run(None, {name: x})
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats, np.concatenate([o.ravel() for o in out])


def run_c(plan, weights, input_path, variant, batch, threads, runs, warmup, extra=()):
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "runs.csv")
        out_path = os.path.join(tmp, "out.bin")
        cmd = [RUNNER, "--plan", plan, "--weights", weights, "--input", input_path, "--output", out_path,
               "--csv", csv_path, "--variant", variant, "--T", str(threads), "--warmup", str(warmup),
               "--runs", str(runs)]
        if batch:
            cmd += ["--batch", str(batch)]
        cmd += list(extra)
        subprocess.run(cmd, check=True)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return rows, np.fromfile(out_path, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="all")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", default="4,1")
    p.add_argument("--raw", default="raw_logs/harness/whole_model.csv")
    p.add_argument("--summary", default="summary/harness/whole_model.csv")
    args = p.parse_args()

    build()
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", RUNNER,
                    os.path.join(os.path.dirname(SRC), "model_runner.c"), "-lm"], check=True)
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE,
                              universal_newlines=True).stdout.strip()
    timestamp = datetime.datetime.now().isoformat()
    models = MODELS if args.models == "all" else args.models.split(",")
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    raw_fields = ["timestamp", "git_hash", "model", "engine", "threads", "run_id", "total_ms"] + CATEGORIES + \
                 ["freq_mhz", "temp_c", "throttle_hex"]
    new_raw = not os.path.exists(args.raw) or os.path.getsize(args.raw) == 0
    summary = []

    with open(args.raw, "a", newline="") as raw_file:
        raw = csv.DictWriter(raw_file, fieldnames=raw_fields)
        if new_raw:
            raw.writeheader()
        for model in models:
            plan, weights, onnx_path, shape = export_plan(model)
            input_path = os.path.join(PLAN_DIR, f"{model}.input")
            np.random.RandomState(0).randn(1, *shape).astype(np.float32).tofile(input_path)
            for T in (int(t) for t in args.threads.split(",")):
                ort_lats, ort_out = in_own_process(run_ort, onnx_path, input_path, shape, T, args.runs, args.warmup)
                for i, lat in enumerate(ort_lats):
                    raw.writerow({"timestamp": timestamp, "git_hash": git_hash, "model": model,
                                  "engine": "onnxruntime", "threads": T, "run_id": i, "total_ms": lat})
                results = [("onnxruntime", ort_lats, 0.0, {})]
                print(f"{model} T={T} onnxruntime median={st.median(ort_lats):.1f}ms", flush=True)
                for name, variant, batch, extra in ENGINES:
                    rows, out = run_c(plan, weights, input_path, variant, batch, T, args.runs, args.warmup, extra)
                    if out.size != ort_out.size:
                        raise RuntimeError(f"{model} {name}: output size {out.size} != {ort_out.size}")
                    err = float(np.max(np.abs(out - ort_out)) / np.max(np.abs(ort_out)))
                    lats = [float(r["total_ms"]) for r in rows]
                    for r in rows:
                        raw.writerow({"timestamp": timestamp, "git_hash": git_hash, "model": model,
                                      "engine": name, "threads": T, **r})
                    cats = {c: round(st.median(float(r[c]) for r in rows), 3) for c in CATEGORIES}
                    results.append((name, lats, err, cats))
                    flags = sorted({r["throttle_hex"] for r in rows} - {"0x0"})
                    print(f"{model} T={T} {name} median={st.median(lats):.1f}ms err={err:.2e} {cats}"
                          + (f" THROTTLED {flags}" if flags else ""), flush=True)
                raw_file.flush()
                for name, lats, err, cats in results:
                    less = (st.median(ort_lats) - st.median(lats)) / st.median(ort_lats) * 100.0
                    pval = stats.ttest_ind(lats, ort_lats, equal_var=False)[1] if name != "onnxruntime" else None
                    summary.append({
                        "model": model, "engine": name, "threads": T, "n": len(lats),
                        "median_ms": round(st.median(lats), 3), "mean_ms": round(st.mean(lats), 3),
                        "std_ms": round(st.stdev(lats), 3),
                        "pct_less_time_than_ort": round(less, 2), "p_vs_ort": "" if pval is None else f"{pval:.2e}",
                        "output_max_rel_err_vs_ort": f"{err:.2e}", **{c: cats.get(c, "") for c in CATEGORIES},
                    })
            os.remove(weights)

    with open(args.summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
