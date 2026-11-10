"""Temporary: median ms of single 3x3 layers for a grid of harness options.
    python3 benchmarks/sc_sweep.py "SHAPES" "OPTSETS" [T] [RUNS]
SHAPES: C@HW,... (Cin=Cout=C, output HW x HW); OPTSETS: '|'-separated extra harness args."""
import os, subprocess, sys, statistics as st, csv, tempfile
shapes = [s.split("@") for s in sys.argv[1].split(",")]
optsets = sys.argv[2].split("|")
T = sys.argv[3] if len(sys.argv) > 3 else "1"
runs = sys.argv[4] if len(sys.argv) > 4 else "9"
binary = sys.argv[5] if len(sys.argv) > 5 else "build/bench_harness"
for c, hw in shapes:
    line = f"{c}@{hw}:"
    for opts in optsets:
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            cmd = [binary, "--T", T, "--H", str(int(hw) + 2), "--W", str(int(hw) + 2), "--Cin", c, "--Cout", c,
                   "--warmup", "2", "--runs", runs, "--out", f.name] + opts.split()
            subprocess.run(cmd, check=True)
            ms = [float(r[12]) for r in csv.reader(open(f.name))]
        g = 2.0 * int(hw) ** 2 * int(c) ** 2 * 9 / (st.median(ms) / 1e3) / 1e9
        line += f"  [{opts}] {st.median(ms):.2f}ms {g:.1f}GF"
    print(line, flush=True)
