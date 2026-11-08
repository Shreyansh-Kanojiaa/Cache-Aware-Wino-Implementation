/*
 * Laptop build of the int8 engine: HyperRAM, L2 and L1 are arrays, the cluster DMA is memcpy, cores
 * run one after another. Checks every layer against qref.py's outputs (MODEL.layers8).
 *   gcc -O2 -DNET='"models/net8_mv2.h"' -o host8 host8.c && ./host8 mv2
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include NET

#ifndef L1_KB
#define L1_KB 52
#endif
#ifndef L2_KB
#define L2_KB 400
#endif

static uint8_t *RAM;
static FILE *WF, *UF;
static uint8_t *act_l2, *w_l2, *u_l2;
static void ram_rd(uint32_t off, void *dst, uint32_t n) {
    if (act_l2) { memcpy(dst, act_l2 + off, n); return; }
    if (off + n > RAM_USED) { fprintf(stderr, "RAM read past the end\n"); exit(3); }
    memcpy(dst, RAM + off, n);
}
static void ram_wr(uint32_t off, const void *src, uint32_t n) {
    if (act_l2) { memcpy(act_l2 + off, src, n); return; }
    if (off + n > RAM_USED) { fprintf(stderr, "RAM write past the end\n"); exit(3); }
    memcpy(RAM + off, src, n);
}
static void fl_rd(uint32_t off, void *dst, uint32_t n) {
    if (w_l2) { memcpy(dst, w_l2 + off, n); return; }
    if (n == 0) { fprintf(stderr, "0-byte flash read\n"); exit(3); }
    fseek(WF, off, SEEK_SET);
    if (fread(dst, 1, n, WF) != n) { fprintf(stderr, "short flash read\n"); exit(3); }
}
static void ul_rd(uint32_t off, void *dst, uint32_t n) {
    if (n == 0) { fprintf(stderr, "0-byte flash read\n"); exit(3); }
    fseek(UF, off, SEEK_SET);
    if (fread(dst, 1, n, UF) != n) { fprintf(stderr, "short U read\n"); exit(3); }
}
static uint32_t now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint32_t)(t.tv_sec * 1000000u + t.tv_nsec / 1000); }

static uint8_t *L1BASE;
static void check_l1(const void *p, int n) {
    if ((const uint8_t *)p < L1BASE || (const uint8_t *)p + n > L1BASE + L1_KB * 1024) { fprintf(stderr, "L1 access outside the buffer\n"); exit(3); }
}
static void dma_in(void *l1, const void *l2, int n) { check_l1(l1, n); memcpy(l1, l2, n); }
static void dma_out2d(void *l2, const void *l1, int size, int stride, int len) {
    check_l1(l1, size);
    for (int i = 0; i < size / len; i++) memcpy((uint8_t *)l2 + (size_t)i * stride, (const uint8_t *)l1 + (size_t)i * len, len);
}
static void cl_run(void (*pre)(void), void (*fn)(int, int), void (*post)(void));

static inline int32_t sdot4(uint32_t a, uint32_t b, int32_t acc, int sa) {
    for (int i = 0; i < 4; i++) {
        int x = sa ? (int8_t)(a >> (8 * i)) : (uint8_t)(a >> (8 * i));
        acc += x * (int8_t)(b >> (8 * i));
    }
    return acc;
}
#define SDOT4U(a, b, acc) sdot4(a, b, acc, 0)
#define SDOT4S(a, b, acc) sdot4(a, b, acc, 1)
#define SDOT2(a, b, acc) ((acc) + (int16_t)(a) * (int16_t)(b) + (int16_t)((a) >> 16) * (int16_t)((b) >> 16))

#include "engine8.h"

static int NCORE = 8;
static void cl_run(void (*pre)(void), void (*fn)(int, int), void (*post)(void)) {
    pre();
    for (int c = 0; c < NCORE; c++) fn(c, NCORE);
    post();
}

static FILE *REF;
static int bad;
static void check_layer(int i) {   /* this layer's output against qref.py's */
    const Layer *L = &LAYERS[i];
    int n = L->c * L->h * L->w * (L->type == L_FC ? 4 : 1);
    uint8_t *ref = malloc(n), *got = malloc(n);
    if (fread(ref, 1, n, REF) != (size_t)n) { fprintf(stderr, "short reference\n"); exit(2); }
    ram_rd(L->out, got, n);
    int diff = 0;
    for (int j = 0; j < n; j++) diff += ref[j] != got[j];
    if (diff && bad++ < 5) printf("layer %d type %d: %d of %d bytes differ\n", i, L->type, diff, n);
    free(ref); free(got);
}

int main(int argc, char **argv) {
    char p[256];
    snprintf(p, sizeof p, "models/%s.w8", argv[1]); WF = fopen(p, "rb");
    snprintf(p, sizeof p, "models/%s.u8", argv[1]); UF = fopen(p, "rb");
    if (getenv("DIRECT")) use_wino8 = 0;
    if (!WF) { fprintf(stderr, "no weights\n"); return 2; }
    RAM = calloc(RAM_USED, 1);
    A2_n = L2_KB * 1024; A2 = malloc(A2_n);
    A1_n = L1_KB * 1024; A1 = L1BASE = malloc(A1_n);
    if (getenv("ACT_L2")) act_l2 = calloc(RAM_USED, 1);   /* test the in-L2 paths */
    if (getenv("W_L2")) { fseek(WF, 0, SEEK_END); long n = ftell(WF); w_l2 = malloc(n); fseek(WF, 0, SEEK_SET); if (fread(w_l2, 1, n, WF) != (size_t)n) return 2; }
    snprintf(p, sizeof p, "models/%s.in8", argv[1]);
    FILE *f = fopen(p, "rb");
    uint8_t *x = malloc(IN_BYTES);
    if (fread(x, 1, IN_BYTES, f) != IN_BYTES) return 2;
    ram_wr(IN_OFF, x, IN_BYTES);
    snprintf(p, sizeof p, "models/%s.layers8", argv[1]);
    REF = fopen(p, "rb");
    after_layer8 = check_layer;
    run_net8();
    if (fail8) return 1;
    printf("{\"model\": \"%s\", \"layers\": %d, \"mismatching_layers\": %d}\n", argv[1], N_LAYERS, bad);
    return bad != 0;
}
