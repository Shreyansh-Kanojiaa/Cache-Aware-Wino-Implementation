#!/usr/bin/env python3
"""Check that the TFLite and XNNPACK baselines run the way engine_comparison.py claims.

    python3 benchmarks/verify_baselines.py [MODELS] [--xnnpack-dir DIR]

1. Build: how the standalone XNNPACK was compiled (CMake build type and flags).
2. TFLite delegation: per model, how many graph nodes the XNNPACK delegate took and which
   ops (if any) stayed on TFLite's own kernels.
3. TFLite with vs without the delegate: proves the delegate is what gets timed.
4. Threads actually used, at T=1 and T=4, for TFLite, standalone XNNPACK and our runner:
   OS threads alive during inference, and process CPU time / wall time (about how many
   cores were busy; spinning workers count as busy).
"""
import argparse
import os
import resource
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.engine_comparison import XNN_RUNNER, convert_tflite
from benchmarks.whole_model import PLAN_DIR, RUNNER, export_plan, in_own_process


def cmake_build(xnn_dir):
    cache = os.path.join(xnn_dir, "build", "CMakeCache.txt")
    if not os.path.exists(cache):
        return f"no {cache}"
    keep = ("CMAKE_BUILD_TYPE:", "CMAKE_C_FLAGS:", "CMAKE_C_FLAGS_RELEASE:", "CMAKE_CXX_FLAGS_RELEASE:",
            "XNNPACK_ENABLE_ASSEMBLY:", "XNNPACK_ENABLE_ARM_FP16_VECTOR:")
    return "; ".join(l.strip() for l in open(cache) if l.startswith(keep))


def tflite_interp(path, threads, delegate=True):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    kind = OpResolverType.AUTO if delegate else OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    it = Interpreter(model_path=path, num_threads=threads, experimental_op_resolver_type=kind)
    it.allocate_tensors()
    return it


def delegation(path):
    """Delegate partitions: one means the graph runs in XNNPACK as one piece; more mean ops
    between them stayed on TFLite's kernels. (The node list keeps the replaced nodes too, so
    which ops stayed out can't be read from it.)"""
    return [o["op_name"] for o in tflite_interp(path, 4)._get_ops_details()].count("DELEGATE")


def tflite_threads(path, shape, threads, runs, delegate=True):
    it = tflite_interp(path, threads, delegate)
    inp = it.get_input_details()[0]
    x = np.random.RandomState(0).randn(*inp["shape"]).astype(np.float32)
    it.set_tensor(inp["index"], x)
    for _ in range(2):
        it.invoke()
    n_threads = len(os.listdir("/proc/self/task"))
    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    lats = []
    for _ in range(runs):
        a = time.perf_counter()
        it.invoke()
        lats.append((time.perf_counter() - a) * 1000)
    wall, r1 = time.perf_counter() - t0, resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return st.median(lats), n_threads, cpu / wall


def runner_threads(binary, plan, weights, inp, threads, runs, extra=()):
    """Run a C runner, sample its thread count while it runs, CPU/wall from wait4."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "r.csv")
        cmd = [binary, "--plan", plan, "--weights", weights, "--input", inp, "--output", os.path.join(tmp, "o.bin"),
               "--csv", csv_path, "--T", str(threads), "--warmup", "2", "--runs", str(runs)] + list(extra)
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        peak = [0]

        def sample():
            while p.poll() is None:
                try:
                    peak[0] = max(peak[0], len(os.listdir(f"/proc/{p.pid}/task")))
                except FileNotFoundError:
                    break
                time.sleep(0.005)
        th = threading.Thread(target=sample)
        th.start()
        _, status, ru = os.wait4(p.pid, 0)
        wall = time.perf_counter() - t0
        th.join()
        if status != 0:
            raise RuntimeError(f"{cmd[0]} failed")
        import csv
        lats = [float(r["total_ms"]) for r in csv.DictReader(open(csv_path))]
        # CPU/wall covers setup too (weight reordering is single-threaded), so it understates
        return st.median(lats), peak[0], (ru.ru_utime + ru.ru_stime) / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="?", default="mobilenet_v2,resnet18,resnet34,vgg16,yolov8n,alexnet")
    ap.add_argument("--timed", default="mobilenet_v2,resnet18,vgg16")
    ap.add_argument("--xnnpack-dir", default=os.path.expanduser("~/third_party/XNNPACK"))
    args = ap.parse_args()

    import ai_edge_litert
    print(f"LiteRT {ai_edge_litert.__version__}; standalone XNNPACK build: {cmake_build(args.xnnpack_dir)}", flush=True)

    print("\n== TFLite: nodes taken by the XNNPACK delegate", flush=True)
    for m in args.models.split(","):
        _, _, onnx_path, _ = export_plan(m)
        print(f"{m}: {in_own_process(delegation, convert_tflite(m, onnx_path))} delegate partition(s)", flush=True)

    print("\n== Threads in use (median ms | OS threads during inference | CPU time / wall time)", flush=True)
    for m in filter(None, args.timed.split(",")):
        plan, weights, onnx_path, shape = export_plan(m)
        inp = os.path.join(PLAN_DIR, f"{m}.input")
        np.random.RandomState(0).randn(1, *shape).astype(np.float32).tofile(inp)
        tfl = convert_tflite(m, onnx_path)
        runs = 3 if m == "vgg16" else 10
        for T in (1, 4):
            for name, res in (
                # each TFLite measurement in a fresh process, so no earlier interpreter's threads are counted
                ("tflite", lambda: in_own_process(tflite_threads, tfl, shape, T, runs)),
                ("tflite_no_delegate", lambda: in_own_process(tflite_threads, tfl, shape, T, max(2, runs // 3), False)),
                ("xnnpack", lambda: runner_threads(XNN_RUNNER, plan, weights, inp, T, runs)),
                ("ours", lambda: runner_threads(RUNNER, plan, weights, inp, T, runs, ["--variant", "v6_freq"])),
            ):
                ms, n_thr, ratio = res()
                print(f"{m} T={T} {name:19s} {ms:8.1f} ms | {n_thr:3d} threads | cpu/wall {ratio:.2f}", flush=True)
        os.remove(weights)


if __name__ == "__main__":
    main()
