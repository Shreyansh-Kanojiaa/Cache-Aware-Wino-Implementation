"""Run kernels through the OpenMP C harness (src/bench_harness.c).

The NumPy functions in src/fused_winograd_kernel.py are the same arithmetic for
fused and non-fused, so they can't compare the two. These numbers can.
"""
import csv
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "bench_harness.c")
BIN = os.path.join(ROOT, "build", "bench_harness")
FIELDS = ["board", "git_hash", "cell_id", "variant", "m", "T", "H", "W", "Cin", "Cout",
          "warmup_state", "rep_idx", "latency_ms", "implied_gflops", "freq_mhz", "temp_c",
          "throttle_hex", "executed_path"]

_ready = False


def build():
    """Compile for this board and run the C selftest. No fallback: without
    -fopenmp, num_threads(T) is ignored and every T silently runs on one core."""
    global _ready
    if _ready:
        return
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", BIN, SRC, "-lm"], check=True)
    subprocess.run([BIN, "--selftest"], check=True, stdout=subprocess.DEVNULL)
    _ready = True


def conv_output_size(size, stride=1, padding=1):
    """Output size of a 3x3 conv."""
    return (size + 2 * padding - 3) // stride + 1


def measure(variant, m, threads, h_out, w_out, c_in, c_out, runs, warmup,
            board="rpi4", cell_id="cell", raw_path=None, stride=1, batch=None):
    """Run one cell and return its latencies (ms). h_out/w_out are output sizes;
    the harness does a valid conv, so its input is sized to produce exactly those."""
    build()
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    with tempfile.NamedTemporaryFile("r", suffix=".csv") as tmp:
        subprocess.run([BIN, "--board", board, "--git-hash", git_hash or "unknown",
                        "--cell-id", cell_id, "--variant", variant, "--m", str(m),
                        "--T", str(threads), "--stride", str(stride),
                        "--H", str((h_out - 1) * stride + 3), "--W", str((w_out - 1) * stride + 3),
                        "--Cin", str(c_in), "--Cout", str(c_out), "--warmup", str(warmup),
                        "--runs", str(runs), "--out", tmp.name]
                       + (["--batch", str(batch)] if batch else []), check=True)
        rows = list(csv.DictReader(tmp, fieldnames=FIELDS))

    if len(rows) != runs or any(r["executed_path"] != variant for r in rows):
        raise RuntimeError(f"{cell_id}: bad harness output for {variant} ({len(rows)}/{runs} rows)")
    throttled = {r["throttle_hex"] for r in rows} - {"0x0"}
    if throttled:
        print(f"     WARNING {cell_id}: throttle flags {sorted(throttled)}")

    if raw_path:
        os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
        new = not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0
        with open(raw_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerows(rows)
    return [float(r["latency_ms"]) for r in rows]


if __name__ == "__main__":
    assert conv_output_size(224) == 224
    assert conv_output_size(56, stride=2) == 28
    assert conv_output_size(7, stride=2) == 4
    lats = measure("fused", 4, 4, 14, 14, 16, 32, runs=3, warmup=1)
    assert len(lats) == 3 and all(x > 0 for x in lats)
    print("ok", lats)
