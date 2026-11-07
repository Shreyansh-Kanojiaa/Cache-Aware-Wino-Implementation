#!/usr/bin/env python3
"""Compare whole models across engines: ONNX Runtime, our C engine, XNNPACK and ncnn.

Every engine runs the same ONNX graph, weights and input, and each final output is
compared with ONNX Runtime's. Each engine gets its own process. ncnn is set up the
way its own benchncnn tool does it, with pool allocators and powersave off.

Needs XNNPACK built in --xnnpack-dir and the ncnn Python package.
Writes raw_logs/harness/engine_comparison.csv and summary/harness/engine_comparison.csv.
"""
import argparse
import collections
import csv
import datetime
import glob
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.e2e_benchmark import MODELS
from benchmarks.whole_model import ENGINES as C_ENGINES, PLAN_DIR, RUNNER, export_plan, run_c, run_ort
from src.harness_runner import ROOT, SRC, build

XNN_RUNNER = os.path.join(ROOT, "build", "xnnpack_runner")
C_KEEP = {"c_v5_batched", "c_direct_packed"}


def build_xnnpack_runner(xnn_dir):
    """Link against every static library the XNNPACK build produced."""
    libs = sorted(glob.glob(os.path.join(xnn_dir, "build", "**", "*.a"), recursive=True))
    headers = [os.path.join(xnn_dir, "include")] + glob.glob(os.path.join(xnn_dir, "build", "**", "pthreadpool-source", "include"), recursive=True)
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-w", "-c", "-o", XNN_RUNNER + ".o",
                    os.path.join(os.path.dirname(SRC), "xnnpack_runner.c")] + [f"-I{h}" for h in headers], check=True)
    subprocess.run(["g++", "-fopenmp", "-o", XNN_RUNNER, XNN_RUNNER + ".o", "-Wl,--start-group"] + libs +
                   ["-Wl,--end-group", "-lm", "-lpthread", "-ldl"], check=True)


def run_xnnpack(plan, weights, input_path, threads, runs, warmup):
    with tempfile.TemporaryDirectory() as tmp:
        csv_path, out_path = os.path.join(tmp, "runs.csv"), os.path.join(tmp, "out.bin")
        subprocess.run([XNN_RUNNER, "--plan", plan, "--weights", weights, "--input", input_path, "--output", out_path,
                        "--csv", csv_path, "--T", str(threads), "--warmup", str(warmup), "--runs", str(runs)],
                       check=True, stderr=subprocess.DEVNULL)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return rows, np.fromfile(out_path, dtype=np.float32)


def write_ncnn_model(plan, weights, param_path, bin_path):
    """Translate a plan into ncnn's .param and .bin files, inserting Split layers
    because ncnn blobs can have only one consumer."""
    data = np.fromfile(weights, dtype=np.float32)
    shapes, layers, version, fc_outputs = {}, [], collections.Counter(), set()

    def blob(tid):
        return f"t{tid}_{version[tid]}"

    def next_blob(tid):
        version[tid] += 1
        return blob(tid)

    input_blob, output_blob = None, None
    with open(plan) as f:
        for line in f:
            p = line.split()
            op = p[0]
            if op == "TENSOR":
                shapes[int(p[1])] = tuple(int(x) for x in p[2:5])
            elif op == "INPUT":
                c, h, w = shapes[int(p[1])]
                input_blob = blob(int(p[1]))
                layers.append(["Input", [], [input_blob], f"0={w} 1={h} 2={c}", []])
            elif op == "OUTPUT":
                output_blob = blob(int(p[1]))
            elif op == "CONV":
                out, src, cin, cout, k, s, pad, _m, woff, boff = (int(x) for x in p[1:])
                n = cout * cin * k * k
                arrays = [("w", data[woff:woff + n])]
                if boff >= 0:
                    arrays.append(("b", data[boff:boff + cout]))
                i = blob(src)
                layers.append(["Convolution", [i], [next_blob(out)],
                               f"0={cout} 1={k} 3={s} 4={pad} 5={int(boff >= 0)} 6={n}", arrays])
            elif op == "RELU":
                out, src = int(p[1]), int(p[2])
                i = blob(src)
                layers.append(["ReLU", [i], [next_blob(out)], "", []])
            elif op == "MAXPOOL":
                out, src, k, s, pad = (int(x) for x in p[1:])
                i = blob(src)
                layers.append(["Pooling", [i], [next_blob(out)], f"0=0 1={k} 2={s} 3={pad} 5=1", []])
            elif op == "ADD":
                out, a, b = (int(x) for x in p[1:])
                ia, ib = blob(a), blob(b)
                layers.append(["BinaryOp", [ia, ib], [next_blob(out)], "0=0", []])
            elif op == "GAP":
                out, src = int(p[1]), int(p[2])
                i = blob(src)
                layers.append(["Pooling", [i], [next_blob(out)], "0=1 4=1", []])
            elif op == "GEMM":
                out, src, nin, nout, woff, boff = (int(x) for x in p[1:])
                i = blob(src)
                if src not in fc_outputs:  # fully connected outputs are already flat
                    flat = f"flat{out}"
                    layers.append(["Flatten", [i], [flat], "", []])
                    i = flat
                fc_outputs.add(out)
                n = nout * nin
                arrays = [("w", data[woff:woff + n])]
                if boff >= 0:
                    arrays.append(("b", data[boff:boff + nout]))
                layers.append(["InnerProduct", [i], [next_blob(out)], f"0={nout} 1={int(boff >= 0)} 2={n}", arrays])
            else:
                raise ValueError(op)

    # Insert Split for blobs with several consumers, right after their producer
    consumers = collections.defaultdict(list)
    for li, layer in enumerate(layers):
        for bi, name in enumerate(layer[1]):
            consumers[name].append((li, bi))
    final = []
    for li, layer in enumerate(layers):
        final.append(layer)
        for name in layer[2]:
            uses = consumers.get(name, [])
            if len(uses) > 1:
                outs = [f"{name}_s{j}" for j in range(len(uses))]
                for (ci, bi), new in zip(uses, outs):
                    layers[ci][1][bi] = new
                final.append(["Split", [name], outs, "", []])

    blobs = {b for layer in final for b in layer[1] + layer[2]}
    with open(param_path, "w") as f:
        f.write(f"7767517\n{len(final)} {len(blobs)}\n")
        for li, (kind, ins, outs, params, _) in enumerate(final):
            f.write(f"{kind} {kind.lower()}_{li} {len(ins)} {len(outs)} " + " ".join(ins + outs) + f" {params}\n")
    with open(bin_path, "wb") as f:
        for _, _, _, _, arrays in final:
            for kind, array in arrays:
                if kind == "w":
                    f.write(np.zeros(1, dtype=np.uint32).tobytes())  # flag 0 means raw float32
                array.tofile(f)
    return input_blob, output_blob


def run_ncnn(param_path, bin_path, input_blob, output_blob, input_path, shape, threads, runs, warmup, winograd):
    import ncnn

    ncnn.set_cpu_powersave(0)
    ncnn.set_omp_dynamic(0)
    ncnn.set_omp_num_threads(threads)
    blob_pool = ncnn.UnlockedPoolAllocator()
    workspace_pool = ncnn.PoolAllocator()
    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.lightmode = True
    net.opt.use_winograd_convolution = winograd
    net.opt.blob_allocator = blob_pool
    net.opt.workspace_allocator = workspace_pool
    if net.load_param(param_path) != 0 or net.load_model(bin_path) != 0:
        raise RuntimeError("ncnn failed to load the model")
    x = np.fromfile(input_path, dtype=np.float32).reshape(shape)
    mat = ncnn.Mat(x)

    def once():
        ex = net.create_extractor()
        ex.input(input_blob, mat)
        ret, out = ex.extract(output_blob)
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed: {ret}")
        return out

    for _ in range(warmup):
        once()
    lats = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = once()
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats, np.array(out).ravel().copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="all")
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", default="4,1")
    p.add_argument("--xnnpack-dir", default=os.path.expanduser("~/third_party/XNNPACK"))
    p.add_argument("--raw", default="raw_logs/harness/engine_comparison.csv")
    p.add_argument("--summary", default="summary/harness/engine_comparison.csv")
    args = p.parse_args()

    build()
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", RUNNER,
                    os.path.join(os.path.dirname(SRC), "model_runner.c"), "-lm"], check=True)
    build_xnnpack_runner(args.xnnpack_dir)

    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE,
                              universal_newlines=True).stdout.strip()
    xnn_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=args.xnnpack_dir, stdout=subprocess.PIPE,
                              universal_newlines=True).stdout.strip()
    import ncnn
    versions = f"xnnpack={xnn_hash};ncnn={ncnn.__version__}"
    print(versions, flush=True)
    timestamp = datetime.datetime.now().isoformat()
    models = MODELS if args.models == "all" else args.models.split(",")
    ctx = mp.get_context("spawn")
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    fields = ["timestamp", "git_hash", "versions", "model", "engine", "threads", "run_id", "total_ms",
              "freq_mhz", "temp_c", "throttle_hex"]
    new_raw = not os.path.exists(args.raw) or os.path.getsize(args.raw) == 0
    summary = []

    with open(args.raw, "a", newline="") as raw_file:
        raw = csv.DictWriter(raw_file, fieldnames=fields)
        if new_raw:
            raw.writeheader()
        for model in models:
            plan, weights, onnx_path, shape = export_plan(model)
            input_path = os.path.join(PLAN_DIR, f"{model}.input")
            np.random.RandomState(0).randn(1, *shape).astype(np.float32).tofile(input_path)
            param_path = os.path.join(PLAN_DIR, f"{model}.param")
            bin_path = os.path.join(PLAN_DIR, f"{model}.ncnn.bin")
            in_blob, out_blob = write_ncnn_model(plan, weights, param_path, bin_path)

            for T in (int(t) for t in args.threads.split(",")):
                results = []

                def record(name, lats, out, rows=None):
                    err = float(np.max(np.abs(out - reference)) / np.max(np.abs(reference)))
                    for i, lat in enumerate(lats):
                        extra = {k: rows[i][k] for k in ("freq_mhz", "temp_c", "throttle_hex")} if rows else {}
                        raw.writerow({"timestamp": timestamp, "git_hash": git_hash, "versions": versions, "model": model,
                                      "engine": name, "threads": T, "run_id": i, "total_ms": lat, **extra})
                    results.append((name, lats, err))
                    print(f"{model} T={T} {name} median={st.median(lats):.1f}ms err={err:.2e}", flush=True)

                with ctx.Pool(1) as pool:
                    ort_lats, reference = pool.apply(run_ort, (onnx_path, input_path, shape, T, args.runs, args.warmup))
                record("onnxruntime", ort_lats, reference)
                for name, variant, batch in C_ENGINES:
                    if name in C_KEEP:
                        rows, out = run_c(plan, weights, input_path, variant, batch, T, args.runs, args.warmup)
                        record(name, [float(r["total_ms"]) for r in rows], out, rows)
                rows, out = run_xnnpack(plan, weights, input_path, T, args.runs, args.warmup)
                record("xnnpack", [float(r["total_ms"]) for r in rows], out, rows)
                for name, winograd in (("ncnn", True), ("ncnn_no_winograd", False)):
                    with ctx.Pool(1) as pool:
                        lats, out = pool.apply(run_ncnn, (param_path, bin_path, in_blob, out_blob, input_path, shape, T,
                                                          args.runs, args.warmup, winograd))
                    record(name, lats, out)
                raw_file.flush()

                ours = next(r[1] for r in results if r[0] == "c_v5_batched")
                for name, lats, err in results:
                    pval = stats.ttest_ind(ours, lats, equal_var=False)[1] if name != "c_v5_batched" else None
                    summary.append({
                        "model": model, "engine": name, "threads": T, "n": len(lats),
                        "median_ms": round(st.median(lats), 3), "mean_ms": round(st.mean(lats), 3),
                        "std_ms": round(st.stdev(lats), 3),
                        "v5_less_time_pct": round((st.median(lats) - st.median(ours)) / st.median(lats) * 100.0, 2),
                        "p_v5_vs_this": "" if pval is None else f"{pval:.2e}",
                        "output_max_rel_err_vs_ort": f"{err:.2e}", "versions": versions,
                    })
            for path in (weights, bin_path):
                os.remove(path)

    with open(args.summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
