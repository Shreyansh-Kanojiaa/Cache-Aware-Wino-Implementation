# GAP8 layer study

Direct vs Winograd 3x3 stride-1 convolutions on a GAPuino (GAP8, 50 MHz, no FPU), for
ResNet-8's three layer shapes. float32 runs in software; int8 accumulates in int32 and its
F(2,3) Winograd uses integer transforms scaled by 2, so it equals direct exactly. All data
is read from L2; each cluster core's Winograd scratch is in L1. `board_results.jsonl` holds
one line per run with cycles and the accuracy check.

Build and run inside the `gap8` distrobox from the UROP gap8 project (`source env.sh`):

    make clean all run platform=board            # or ./run_board.sh from the host
    make clean all run platform=board EXTRA=-DQUICK   # one 8x8 layer, every path

Not yet done: SIMD int8 kernels, L1 tiling with the cluster DMA, F(4,3) for 64 channels
(its float32 weights, 576 KB, don't fit L2).
