#!/usr/bin/env python3
"""
T5 — End-to-end CacheWinograd benchmarking.

Runs each 3x3 conv layer through the C kernels in src/bench_harness.c (OpenMP,
--threads cores), then sums the per-layer times for each run.

METHODOLOGY NOTE:
  Winograd and direct numbers are layer-time aggregates of 3x3 convs measured
  separately, with each layer's own warmup. They exclude every other op (1x1,
  7x7 and 11x11 convs, pooling, FC), and cache effects between layers. ORT and
  TVM ARE full-model single-process runs, so those comparisons are not like for
  like.

  Stride-2 layers can't use Winograd, so all kernels run them as neon_direct
  with the same output size.

Models: AlexNet, ResNet-18, ResNet-34, VGG16
Baselines: nonfused_winograd, neon_direct, onnxruntime, tvm, autotvm
n=50 minimum per model per platform.

Logs raw per-run latency to raw_logs/harness/e2e_<model>_<platform>.csv and
per-layer rows to raw_logs/harness/e2e_<model>_<platform>_layers.csv.
Produces summary/harness/e2e_comparison_T<threads>.csv.
"""
import os
import sys
import time
import datetime
import argparse
import csv
import platform as platform_mod
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.harness_runner import conv_output_size, measure
from src.cache_adaptive_autotiler import CacheAdaptiveAutotiler
from src.runtime_cache_probe import build_platform_descriptor

# Optional imports
try:
    import torch
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import tvm
    from tvm import relay
    from tvm.contrib import graph_executor
    TVM_AVAILABLE = True
except ImportError:
    TVM_AVAILABLE = False

MODELS = ["alexnet", "resnet18", "resnet34", "vgg16"]

MODEL_DISPLAY = {
    "alexnet": "AlexNet",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "vgg16": "VGG16",
}


def _detect_platform_name():
    """Detect a human-readable platform name."""
    desc = build_platform_descriptor()
    os_name = desc.get("os", "unknown")
    cpu = str(desc.get("cpu_model", "")).lower()
    if os_name == "Darwin":
        return "macos_apple_silicon"
    elif os_name == "Linux":
        if "tegra" in cpu or "cortex-a57" in cpu:
            return "jetson_nano"
        elif "bcm2711" in cpu or "raspberry" in cpu:
            return "rpi4"
        elif "bcm2712" in cpu:
            return "rpi5"
    return os_name.lower()


def _get_conv3x3_layers(model_name):
    """Extract all 3×3 conv layer shapes from a model.

    Returns list of dicts: {c_in, c_out, h_in, w_in, stride, padding}
    """
    if not TORCH_AVAILABLE:
        # Hardcoded layer shapes as fallback
        return _get_conv3x3_layers_hardcoded(model_name)

    model_fn = {
        "alexnet": models.alexnet,
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "vgg16": models.vgg16,
    }
    if model_name not in model_fn:
        return []

    model = model_fn[model_name](weights=None)
    model.eval()

    layers = []
    hooks = []

    def _hook_fn(module, input, output):
        if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3):
            layers.append({
                "c_in": module.in_channels,
                "c_out": module.out_channels,
                "h_in": input[0].shape[2],
                "w_in": input[0].shape[3],
                "stride": module.stride[0],
                "padding": module.padding[0],
            })

    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (3, 3):
            hooks.append(mod.register_forward_hook(_hook_fn))

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        model(dummy)

    for h in hooks:
        h.remove()

    return layers


def _get_conv3x3_layers_hardcoded(model_name):
    """Hardcoded 3×3 conv layer shapes for common models."""
    # These are the standard shapes from torchvision models at 224×224 input
    if model_name == "alexnet":
        # AlexNet has no 3×3 convs in its standard definition (5×5, 3×3, 3×3, 3×3)
        return [
            {"c_in": 192, "c_out": 384, "h_in": 13, "w_in": 13, "stride": 1, "padding": 1},
            {"c_in": 384, "c_out": 256, "h_in": 13, "w_in": 13, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 256, "h_in": 13, "w_in": 13, "stride": 1, "padding": 1},
        ]
    elif model_name == "resnet18":
        return [
            {"c_in": 64, "c_out": 64, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 64, "c_out": 64, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 64, "c_out": 64, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 64, "c_out": 64, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 64, "c_out": 128, "h_in": 28, "w_in": 28, "stride": 2, "padding": 1},
            {"c_in": 128, "c_out": 128, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1},
            {"c_in": 128, "c_out": 128, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1},
            {"c_in": 128, "c_out": 128, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1},
            {"c_in": 128, "c_out": 256, "h_in": 7, "w_in": 7, "stride": 2, "padding": 1},
            {"c_in": 256, "c_out": 256, "h_in": 7, "w_in": 7, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 256, "h_in": 7, "w_in": 7, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 256, "h_in": 7, "w_in": 7, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 512, "h_in": 4, "w_in": 4, "stride": 2, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 4, "w_in": 4, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 4, "w_in": 4, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 4, "w_in": 4, "stride": 1, "padding": 1},
        ]
    elif model_name == "resnet34":
        layers = []
        # Block 1: 3 layers of 64->64 at 56x56
        for _ in range(6):
            layers.append({"c_in": 64, "c_out": 64, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1})
        # Block 2: first is stride-2, then 3 more
        layers.append({"c_in": 64, "c_out": 128, "h_in": 28, "w_in": 28, "stride": 2, "padding": 1})
        for _ in range(7):
            layers.append({"c_in": 128, "c_out": 128, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1})
        # Block 3
        layers.append({"c_in": 128, "c_out": 256, "h_in": 7, "w_in": 7, "stride": 2, "padding": 1})
        for _ in range(11):
            layers.append({"c_in": 256, "c_out": 256, "h_in": 7, "w_in": 7, "stride": 1, "padding": 1})
        # Block 4
        layers.append({"c_in": 256, "c_out": 512, "h_in": 4, "w_in": 4, "stride": 2, "padding": 1})
        for _ in range(5):
            layers.append({"c_in": 512, "c_out": 512, "h_in": 4, "w_in": 4, "stride": 1, "padding": 1})
        return layers
    elif model_name == "vgg16":
        return [
            {"c_in": 3, "c_out": 64, "h_in": 224, "w_in": 224, "stride": 1, "padding": 1},
            {"c_in": 64, "c_out": 64, "h_in": 224, "w_in": 224, "stride": 1, "padding": 1},
            {"c_in": 64, "c_out": 128, "h_in": 112, "w_in": 112, "stride": 1, "padding": 1},
            {"c_in": 128, "c_out": 128, "h_in": 112, "w_in": 112, "stride": 1, "padding": 1},
            {"c_in": 128, "c_out": 256, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 256, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 256, "h_in": 56, "w_in": 56, "stride": 1, "padding": 1},
            {"c_in": 256, "c_out": 512, "h_in": 28, "w_in": 28, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 28, "w_in": 28, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 28, "w_in": 28, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1},
            {"c_in": 512, "c_out": 512, "h_in": 14, "w_in": 14, "stride": 1, "padding": 1},
        ]
    return []


def benchmark_harness_e2e(model_name, variant, n_runs, warmup, conv_layers, threads,
                          platform_name):
    """Sum per-layer harness latencies for each run."""
    tiler = CacheAdaptiveAutotiler()
    raw_path = os.path.join("raw_logs", "harness", f"e2e_{model_name}_{platform_name}_layers.csv")
    totals = np.zeros(n_runs)
    for i, layer in enumerate(conv_layers):
        stride = layer.get("stride", 1)
        padding = layer.get("padding", 1)
        # ponytail: stride-2 is timed as a stride-1 direct conv with the same output
        # size, which matches its work but not its memory stride. Add stride to the
        # harness if these layers ever matter.
        use = "neon_direct" if stride > 1 else variant
        td = tiler.select_best_tile(layer["c_in"], layer["c_out"])["selected_tile"]["tile"]
        totals += measure(use, td - 2, threads,
                          conv_output_size(layer["h_in"], stride, padding),
                          conv_output_size(layer["w_in"], stride, padding),
                          layer["c_in"], layer["c_out"], n_runs, warmup, board=platform_name,
                          cell_id=f"{model_name}_L{i}_{variant}_T{threads}", raw_path=raw_path)
    return totals.tolist()


def benchmark_onnxruntime_e2e(model_name, n_runs, warmup, threads):
    """Full-model ONNX Runtime baseline."""
    if not ORT_AVAILABLE:
        return None

    onnx_path = os.path.join("models", "onnx", f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = f"{model_name}.onnx"
    if not os.path.exists(onnx_path):
        # Try to export
        if TORCH_AVAILABLE:
            _export_onnx(model_name, onnx_path)
        else:
            return None

    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"     [onnxruntime] Failed to load {onnx_path}: {e}")
        return None

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Handle dynamic shapes
    actual_shape = [d if isinstance(d, int) else 1 for d in input_shape]
    input_data = np.random.randn(*actual_shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: input_data})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: input_data})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def _export_onnx(model_name, path):
    """Export a torchvision model to ONNX."""
    model_fn = {
        "alexnet": models.alexnet,
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "vgg16": models.vgg16,
    }
    if model_name not in model_fn:
        return
    model = model_fn[model_name](weights=None).eval()
    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(model, dummy, path, opset_version=13,
                       input_names=["input"], output_names=["output"])


def benchmark_tvm_e2e(model_name, n_runs, warmup, tvm_target=None, autotvm=False):
    """Full-model TVM baseline."""
    if not TVM_AVAILABLE or not ORT_AVAILABLE:
        return None

    onnx_path = os.path.join("models", "onnx", f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = f"{model_name}.onnx"
    if not os.path.exists(onnx_path):
        return None

    if tvm_target is None:
        machine = platform_mod.machine().lower()
        if machine in ("arm64", "aarch64"):
            tvm_target = "llvm -mcpu=apple-m1" if sys.platform == "darwin" else "llvm -mattr=+neon"
        else:
            tvm_target = "llvm"

    try:
        onnx_model = onnx.load(onnx_path)
        mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": [1, 3, 224, 224]})
        if autotvm and TVM_AVAILABLE:
            from tvm import autotvm as _autotvm
            import tempfile
            tasks = _autotvm.task.extract_from_program(
                mod["main"], target=tvm.target.Target(tvm_target), params=params)
            if tasks:
                with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as lf:
                    log_path = lf.name
                measure_option = _autotvm.measure_option(
                    builder=_autotvm.LocalBuilder(),
                    runner=_autotvm.LocalRunner(number=1, repeat=1, min_repeat_ms=0))
                try:
                    for task in reversed(tasks):
                        try:
                            tuner = _autotvm.tuner.XGBTuner(task)
                        except Exception:
                            tuner = _autotvm.tuner.RandomTuner(task)
                        tuner.tune(n_trial=min(32, len(task.config_space)),
                                   measure_option=measure_option,
                                   callbacks=[_autotvm.callback.log_to_file(log_path)])
                    with _autotvm.apply_history_best(log_path):
                        with tvm.transform.PassContext(opt_level=3):
                            lib = relay.build(mod, target=tvm_target, params=params)
                finally:
                    if os.path.exists(log_path):
                        os.remove(log_path)
            else:
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target=tvm_target, params=params)
        else:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=tvm_target, params=params)
        dev = tvm.cpu(0)
        module = graph_executor.GraphModule(lib["default"](dev))
    except Exception as e:
        backend_name = "autotvm" if autotvm else "tvm"
        print(f"     [{backend_name}] Build failed for {model_name}: {e}")
        return None

    input_data = np.random.randn(1, 3, 224, 224).astype("float32")
    module.set_input("input", tvm.nd.array(input_data))

    for _ in range(warmup):
        module.run()

    latencies = []
    for _ in range(n_runs):
        t0 = time.monotonic()
        module.run()
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def run_e2e(model_list=None, n_runs=50, warmup=10, platform_name=None, tvm_target=None, threads=4):
    """Run full E2E benchmarks."""
    from scipy import stats as sp_stats

    if model_list is None:
        model_list = MODELS
    if platform_name is None:
        platform_name = _detect_platform_name()

    os.environ["TVM_NUM_THREADS"] = str(threads)
    timestamp = datetime.datetime.now().isoformat()
    all_raw_rows = []
    summary_rows = []

    for model_name in model_list:
        display = MODEL_DISPLAY.get(model_name, model_name)
        print(f"\n{'='*60}")
        print(f"[T5] E2E Benchmark: {display} on {platform_name}")
        print(f"{'='*60}")

        conv_layers = _get_conv3x3_layers(model_name)
        n_3x3 = len([l for l in conv_layers if l.get("stride", 1) == 1])
        print(f"     {len(conv_layers)} conv3x3 layers detected ({n_3x3} stride-1)")

        baselines = {}

        # 1. CacheWinograd (fused) — layer-time aggregate, not a single trace
        print(f"     Running CacheWinograd fused on {threads} thread(s)...")
        cw_lats = benchmark_harness_e2e(model_name, "fused", n_runs, warmup, conv_layers,
                                        threads, platform_name)
        baselines["cachewinograd"] = cw_lats
        cw_mean = float(np.mean(cw_lats))
        print(f"       mean={cw_mean:.2f}ms  [layer-time aggregate, NOT a single e2e trace]")

        # 2. Non-fused Winograd
        print(f"     Running Non-fused Winograd...")
        baselines["nonfused_winograd"] = benchmark_harness_e2e(
            model_name, "unfused", n_runs, warmup, conv_layers, threads, platform_name)

        # 3. Direct convolution
        print(f"     Running direct conv...")
        baselines["neon_direct"] = benchmark_harness_e2e(
            model_name, "neon_direct", n_runs, warmup, conv_layers, threads, platform_name)

        # 4. ONNX Runtime
        print(f"     Running ONNX Runtime...")
        ort_lats = benchmark_onnxruntime_e2e(model_name, n_runs, warmup, threads)
        if ort_lats:
            baselines["onnxruntime"] = ort_lats
        else:
            print(f"       SKIPPED (not available)")

        # 5. TVM (plain)
        print(f"     Running TVM...")
        tvm_lats = benchmark_tvm_e2e(model_name, n_runs, warmup, tvm_target)
        if tvm_lats:
            baselines["tvm"] = tvm_lats
        else:
            print(f"       SKIPPED (not available)")

        # 6. AutoTVM
        print(f"     Running AutoTVM...")
        autotvm_lats = benchmark_tvm_e2e(model_name, n_runs, warmup, tvm_target, autotvm=True)
        if autotvm_lats:
            baselines["autotvm"] = autotvm_lats
        else:
            print(f"       SKIPPED (not available)")

        # Log raw data
        for bl_name, lats in baselines.items():
            if lats is None:
                continue
            for run_id, lat in enumerate(lats):
                all_raw_rows.append({
                    "timestamp": timestamp,
                    "model": model_name,
                    "platform": platform_name,
                    "baseline": bl_name,
                    "threads": threads,
                    "run_id": run_id,
                    "latency_ms": lat,
                })

        # Compute summary statistics
        for bl_name, lats in baselines.items():
            if lats is None:
                continue
            mean_ms = float(np.mean(lats))
            std_ms = float(np.std(lats, ddof=1))
            ci95 = 1.96 * std_ms / np.sqrt(len(lats))

            p_value = "N/A"
            improvement = "N/A"
            if bl_name != "cachewinograd" and cw_lats:
                _, p = sp_stats.ttest_ind(cw_lats, lats, equal_var=False)
                p_value = f"{p:.6e}" if p >= 1e-10 else "< 1e-10"
                improvement = round(((mean_ms - cw_mean) / mean_ms) * 100.0, 2)

            summary_rows.append({
                "model": display,
                "platform": platform_name,
                "baseline": bl_name,
                "threads": threads,
                "mean_ms": round(mean_ms, 4),
                "CI95": round(ci95, 4),
                "std_ms": round(std_ms, 4),
                "p_value": p_value,
                "pct_improvement": improvement,
                "n_runs": len(lats),
            })

        # Write per-model raw log (append)
        raw_path = os.path.join("raw_logs", "harness", f"e2e_{model_name}_{platform_name}.csv")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        model_rows = [r for r in all_raw_rows if r["model"] == model_name]
        if model_rows:
            file_exists = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
            with open(raw_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(model_rows)
            print(f"     Wrote {len(model_rows)} rows to {raw_path}")

    # Write combined summary ONCE after all models complete
    # (accumulate summary_rows across all models, then write once
    # so a mid-run failure does not truncate earlier models' data)
    if summary_rows:
        summary_path = os.path.join("summary", "harness", f"e2e_comparison_T{threads}.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[T5] E2E summary written to {summary_path}")
        print("[T5] METHODOLOGY: CacheWinograd E2E = layer-time aggregate; "
              "ORT/TVM = true single-process runs.")


def main():
    parser = argparse.ArgumentParser(description="T5: End-to-end CacheWinograd benchmarking")
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--platform", default=None,
                        help="Platform name override (e.g., jetson_nano, rpi4)")
    parser.add_argument("--tvm-target", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    model_list = MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    run_e2e(model_list=model_list, n_runs=args.runs, warmup=args.warmup,
            platform_name=args.platform, tvm_target=args.tvm_target, threads=args.threads)


if __name__ == "__main__":
    main()
