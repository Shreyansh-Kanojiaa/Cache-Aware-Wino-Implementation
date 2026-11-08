/*
 * Laptop build of the GAP8 engine: HyperRAM is an 8 MB array, the flash files are read with
 * stdio, and the "cores" run one after another. Same float operations in the same order as
 * the chip (build with -ffp-contract=off), so the output hash must match the board's.
 *
 *   gcc -O2 -ffp-contract=off -DNET='"net_resnet8.h"' [-DARENA_KB=n] -o host host.c -lm
 *   ./host MODEL   (reads models/MODEL.{bin,u,input,ref})
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include NET

#ifndef ARENA_KB
#define ARENA_KB 380
#endif

static uint8_t *RAM;
static FILE *WF, *RF, *UF;

static void ram_rd(uint32_t off, void *dst, uint32_t n) {
    if (off + n > (8u << 20)) { fprintf(stderr, "RAM read past 8 MB\n"); exit(3); }
    memcpy(dst, RAM + off, n);
}
static void ram_wr(uint32_t off, const void *src, uint32_t n) {
    if (off + n > (8u << 20)) { fprintf(stderr, "RAM write past 8 MB\n"); exit(3); }
    memcpy(RAM + off, src, n);
}
static void rd_file(FILE *f, uint32_t off, void *dst, uint32_t n) {
    if (n == 0) { fprintf(stderr, "0-byte flash read\n"); exit(3); }   /* corrupts L2 on the chip */
    fseek(f, off, SEEK_SET);
    if (fread(dst, 1, n, f) != n) { fprintf(stderr, "short flash read at %u\n", off); exit(3); }
}
static void fl_rd(uint32_t off, void *dst, uint32_t n) { rd_file(WF, off, dst, n); }
static void fl_rd_ref(uint32_t off, void *dst, uint32_t n) { rd_file(RF, off, dst, n); }
static void fl_rd_u(uint32_t off, void *dst, uint32_t n) { rd_file(UF, off, dst, n); }
static void par(void (*fn)(int, int));
static uint32_t now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)(t.tv_sec * 1000000u + t.tv_nsec / 1000);
}

#include "engine.h"

static void par(void (*fn)(int, int)) { for (int c = 0; c < NC; c++) fn(c, NC); }

int main(int argc, char **argv) {
    if (getenv("DIRECT")) use_wino = 0;
    if (argc < 2) { fprintf(stderr, "usage: host MODEL\n"); return 2; }
    RAM = calloc(8u << 20, 1);
    A_n = ARENA_KB * 1024 / 4;
    A = malloc((size_t)A_n * 4);
    if (!getenv("NO_L1")) { L1S_n = NC * (32 + CC) * 36; L1S = malloc((size_t)L1S_n * 4); }
    char p[256];
    snprintf(p, sizeof p, "models/%s.bin", argv[1]); WF = fopen(p, "rb");
    snprintf(p, sizeof p, "models/%s.ref", argv[1]); RF = fopen(p, "rb");
    snprintf(p, sizeof p, "models/%s.u", argv[1]); UF = fopen(p, "rb");
    snprintf(p, sizeof p, "models/%s.input", argv[1]);
    FILE *xf = fopen(p, "rb");
    if (!WF || !RF || !UF || !xf) { fprintf(stderr, "cannot open files\n"); return 2; }
    const Ten *in = &TENS[IN_ID];
    fseek(xf, 0, SEEK_END);
    int n_in = (int)(ftell(xf) / (numel(in) * 4)), out_n = 0;
    for (int i = 0; i < N_OUTS; i++) out_n += numel(&TENS[OUTS[i]]);
    float *x = malloc((size_t)numel(in) * 4);
    for (int img = 0; img < n_in; img++) {
        rd_file(xf, (uint32_t)img * numel(in) * 4, x, numel(in) * 4);
        ram_wr(in->off, x, numel(in) * 4);
        run_net();
        if (fail) return 1;
        printf("{\"input\": %d", img);
        int ref_off = img * out_n;
        for (int i = 0; i < N_OUTS; i++) {
            uint32_t h = 2166136261u; float err = 0, mx = 0;
            const Ten *t = &TENS[OUTS[i]];
            check_output(t, ref_off, &h, &err, &mx);
            ref_off += numel(t);
            printf(", \"out%d_hash\": \"%08x\", \"out%d_err\": %.3g, \"out%d_max\": %.3g", i, h, i, err, i, mx);
        }
        if (N_OUTS == 1 && numel(&TENS[OUTS[0]]) <= 1000) {
            const Ten *t = &TENS[OUTS[0]]; int best = 0;
            ram_rd(t->off, A, numel(t) * 4);
            for (int i = 1; i < numel(t); i++) if (A[i] > A[best]) best = i;
            printf(", \"pred\": %d", best);
        }
        printf("}\n");
    }
    printf("{\"arena_kb\": %d, \"wino\": %d", ARENA_KB, use_wino);
    for (int c = 0; c < N_CAT; c++) printf(", \"%s_mmac\": %.1f", CAT_NAME[c], cat_macs[c] / 1e6);
    printf("}\n");
    return 0;
}
