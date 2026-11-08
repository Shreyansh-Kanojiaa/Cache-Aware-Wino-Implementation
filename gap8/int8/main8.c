/*
 * GAP8 (GAPuino) runner for the int8 engine: weights, input and reference logits in HyperFlash
 * (readfs), activations in HyperRAM, layers staged through L2 and computed in L1 on the 8 cores.
 *   make clean all run platform=board MODEL=resnet8
 */
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram.h"
#include "bsp/ram/hyperram.h"
#include NET

static struct pi_device ram, flash, fs, cl;
static pi_fs_file_t *wf, *xf, *rf, *uf;
static uint32_t ram_base;

/* When activations (or weights) fit L2 next to a working arena, they live there instead of in
 * HyperRAM (or flash), as DORY's all-L2 build does for small networks. */
static uint8_t *act_l2, *w_l2, *u_l2;   /* set before including the engine, which reads them */
static void ram_rd(uint32_t off, void *dst, uint32_t n) {
    if (act_l2) memcpy(dst, act_l2 + off, n);
    else if (n) pi_ram_read(&ram, ram_base + off, dst, n);
}
static void ram_wr(uint32_t off, const void *src, uint32_t n) {
    if (act_l2) memcpy(act_l2 + off, src, n);
    else if (n) pi_ram_write(&ram, ram_base + off, (void *)src, n);
}
static void fl_rd(uint32_t off, void *dst, uint32_t n) {
    if (w_l2) { memcpy(dst, w_l2 + off, n); return; }
    if (!n) return;   /* a 0-byte read corrupts L2 */
    pi_fs_seek(wf, off);
    pi_fs_read(wf, dst, n);
}
static void ul_rd(uint32_t off, void *dst, uint32_t n) {
    if (u_l2) { memcpy(dst, u_l2 + off, n); return; }
    if (!n) return;
    pi_fs_seek(uf, off);
    pi_fs_read(uf, dst, n);
}
static uint32_t now_us(void) { return (uint32_t)pi_time_get_us(); }

/* cluster DMA, issued by core 0. L2->L1 only as 1D copies (2D loads return wrong data on this board) */
#define DMA_MAX 32768
static void dma_in(void *l1, const void *l2, int n) {
    pi_cl_dma_cmd_t c;
    for (int o = 0; o < n; o += DMA_MAX) {
        int len = n - o < DMA_MAX ? n - o : DMA_MAX;
        pi_cl_dma_cmd((uint32_t)l2 + o, (uint32_t)l1 + o, len, PI_CL_DMA_DIR_EXT2LOC, &c);
        pi_cl_dma_cmd_wait(&c);
    }
}
static void dma_out2d(void *l2, const void *l1, int size, int stride, int len) {
    pi_cl_dma_cmd_t c;
    if (stride == len) {
        for (int o = 0; o < size; o += DMA_MAX) {
            int n = size - o < DMA_MAX ? size - o : DMA_MAX;
            pi_cl_dma_cmd((uint32_t)l2 + o, (uint32_t)l1 + o, n, PI_CL_DMA_DIR_LOC2EXT, &c);
            pi_cl_dma_cmd_wait(&c);
        }
        return;
    }
    int lines = DMA_MAX / len;   /* whole lines per command */
    for (int l0 = 0; l0 < size / len; l0 += lines) {
        int nl = size / len - l0 < lines ? size / len - l0 : lines;
        pi_cl_dma_cmd_2d((uint32_t)l2 + l0 * stride, (uint32_t)l1 + l0 * len, nl * len, stride, len, PI_CL_DMA_DIR_LOC2EXT, &c);
        pi_cl_dma_cmd_wait(&c);
    }
}
static void cl_run(void (*pre)(void), void (*fn)(int, int), void (*post)(void));
#define SDOT4U(a, b, acc) __builtin_pulp_sdotusp4((v4u)(a), (v4s)(b), (acc))
#define SDOT4S(a, b, acc) __builtin_pulp_sdotsp4((v4s)(a), (v4s)(b), (acc))
#define SDOT2(a, b, acc) __builtin_pulp_sdotsp2((v2s)(a), (v2s)(b), (acc))

#include "engine8.h"

static void (*g_pre)(void), (*g_fn)(int, int), (*g_post)(void);
static uint32_t cyc_in, cyc_fn, cyc_out, n_tasks;   /* core 0's cycles in each phase */
static void team(void *arg) {
    int c = pi_core_id();
    uint32_t t0 = 0, t1 = 0, t2 = 0;
    if (c == 0) { pi_perf_conf(1 << PI_PERF_CYCLES); pi_perf_reset(); pi_perf_start(); g_pre(); }
    pi_cl_team_barrier();
    if (c == 0) t0 = pi_perf_read(PI_PERF_CYCLES);
    g_fn(c, NC8);
    pi_cl_team_barrier();
    if (c == 0) { t1 = pi_perf_read(PI_PERF_CYCLES); g_post(); t2 = pi_perf_read(PI_PERF_CYCLES); cyc_in += t0; cyc_fn += t1 - t0; cyc_out += t2 - t1; n_tasks++; }
}
static void cl_entry(void *arg) { pi_cl_team_fork(NC8, team, NULL); }
static void cl_run(void (*pre)(void), void (*fn)(int, int), void (*post)(void)) {
    struct pi_cluster_task t;
    g_pre = pre; g_fn = fn; g_post = post;
    pi_cluster_task(&t, cl_entry, NULL);
    t.stack_size = 1536;
    t.slave_stack_size = 1024;
    pi_cluster_send_task_to_cl(&cl, &t);
}

static void run(void) {
    struct pi_hyperram_conf rconf; struct pi_hyperflash_conf fconf; struct pi_readfs_conf fsconf; struct pi_cluster_conf cconf;
    pi_hyperram_conf_init(&rconf); pi_open_from_conf(&ram, &rconf);
    if (pi_ram_open(&ram) || pi_ram_alloc(&ram, &ram_base, RAM_USED)) { printf("FAIL: HyperRAM\n"); pmsis_exit(-1); }
    pi_hyperflash_conf_init(&fconf); pi_open_from_conf(&flash, &fconf);
    if (pi_flash_open(&flash)) { printf("FAIL: flash\n"); pmsis_exit(-2); }
    pi_readfs_conf_init(&fsconf); fsconf.fs.flash = &flash; pi_open_from_conf(&fs, &fsconf);
    if (pi_fs_mount(&fs)) { printf("FAIL: readfs\n"); pmsis_exit(-3); }
    wf = pi_fs_open(&fs, MODEL ".w8", 0); xf = pi_fs_open(&fs, MODEL ".in8", 0); rf = pi_fs_open(&fs, MODEL ".ref8", 0); uf = pi_fs_open(&fs, MODEL ".u8", 0);
    if (!wf || !xf || !rf || !uf) { printf("FAIL: files\n"); pmsis_exit(-4); }
    pi_cluster_conf_init(&cconf); pi_open_from_conf(&cl, &cconf);
    if (pi_cluster_open(&cl)) { printf("FAIL: cluster\n"); pmsis_exit(-5); }
    int big;
    for (big = 440 * 1024; big > 16 * 1024 && !(A2 = pi_l2_malloc(big)); big -= 4096);
    pi_l2_free(A2, big);
    big -= 8192;
#ifndef NO_L2_RESIDENT
    if ((int)RAM_USED + 160 * 1024 <= big) {            /* activations in L2 */
        act_l2 = pi_l2_malloc(RAM_USED); big -= (RAM_USED + 7) & ~7;
    }
    if (act_l2 && (int)wf->size + 128 * 1024 <= big) {  /* weights too, read once here (not timed) */
        w_l2 = pi_l2_malloc(wf->size); big -= (wf->size + 7) & ~7;
        pi_fs_seek(wf, 0);
        pi_fs_read(wf, w_l2, wf->size);
    }
    if (w_l2 && uf->size > 4 && (int)uf->size + 128 * 1024 <= big) {   /* Winograd weights too */
        u_l2 = pi_l2_malloc(uf->size); big -= (uf->size + 7) & ~7;
        pi_fs_seek(uf, 0);
        pi_fs_read(uf, u_l2, uf->size);
    }
#endif
    A2_n = big; A2 = pi_l2_malloc(A2_n);
    /* L1: the largest block, less room for the 8 cluster stacks (1.5 KB + 7 x 1 KB) */
    for (A1_n = 63 * 1024; A1_n > 4096 && !(A1 = pi_cl_l1_malloc(&cl, A1_n)); A1_n -= 1024);
    pi_cl_l1_free(&cl, A1, A1_n); A1_n -= 10 * 1024; A1 = pi_cl_l1_malloc(&cl, A1_n);
    uint64_t macs = 0;
    for (int i = 0; i < N_LAYERS; i++) {
        const Layer *L = &LAYERS[i];
        if (L->type == L_CONV) macs += (uint64_t)L->c * L->h * L->w * L->k * L->k * (L->group == 1 ? L->cin : 1);
        if (L->type == L_FC) macs += (uint64_t)L->cin * L->c;
    }
    printf("{\"engine\": \"ours_int8\", \"model\": \"%s\", \"fc_hz\": %d, \"cl_hz\": %d, \"l2_kb\": %d, \"l1_kb\": %d, \"macs\": %u, \"activations_in\": \"%s\", \"weights_in\": \"%s\"}\n",
           MODEL, pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), A2_n / 1024, A1_n / 1024, (unsigned)macs,
           act_l2 ? "L2" : "HyperRAM", w_l2 ? "L2" : "HyperFlash");
    static uint8_t x[IN_BYTES];
    static int32_t ref[OUT_N], y[OUT_N];
    pi_fs_read(xf, x, IN_BYTES);
    pi_fs_read(rf, ref, OUT_N * 4);
    for (int run = 0; run < 4; run++) {
        use_wino8 = run < 2;
        ram_wr(IN_OFF, x, IN_BYTES);
        cyc_in = cyc_fn = cyc_out = n_tasks = 0;
        uint32_t t0 = now_us();
        run_net8();
        uint32_t us = now_us() - t0;
        if (fail8) pmsis_exit(-6);
        ram_rd(LAYERS[N_LAYERS - 1].out, y, OUT_N * 4);
        int diff = 0, best = 0;
        for (int i = 0; i < OUT_N; i++) { diff += y[i] != ref[i]; if (y[i] > y[best]) best = i; }
        unsigned cyc = us * (pi_freq_get(PI_FREQ_DOMAIN_CL) / 1000000);
        printf("{\"engine\": \"ours_int8\", \"model\": \"%s\", \"run\": %d, \"us\": %u, \"cycles\": %u, \"mac_per_cycle_e2\": %u, \"io_us\": %u, "
               "\"logits_differing\": %d, \"pred\": %d, \"wino\": %d", MODEL, run, us, cyc, (unsigned)(macs * 100 / cyc), io8_us, diff, best, use_wino8);
        for (int c = 0; c < N_C8; c++) printf(", \"%s_us\": %u", C8_NAME[c], cat8_us[c]);
        printf(", \"cl_dma_in_cyc\": %u, \"cl_compute_cyc\": %u, \"cl_dma_out_cyc\": %u, \"cl_tasks\": %u", cyc_in, cyc_fn, cyc_out, n_tasks);
        printf("}\n");
    }
    printf("{\"layer_us\": [");
    for (int i = 0; i < N_LAYERS; i++) printf("%s%u", i ? ", " : "", lay_us[i]);
    printf("]}\n{\"done\": 1}\n");
    pmsis_exit(0);
}

#ifndef FREQ_MHZ
#define FREQ_MHZ 100   /* DORY's generated main runs the FC and cluster at 100 MHz; the comparison uses the same */
#endif
int main(void) {
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_MHZ * 1000000);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_MHZ * 1000000);
    pi_time_wait_us(10000);
    return pmsis_kickoff((void *)run);
}
