/*
 * GAP8 (GAPuino) whole-model runner. Weights, inputs and reference outputs are readfs files in
 * HyperFlash; activations are in HyperRAM; layers stream through one L2 arena and run on the
 * 8-core cluster. Prints one JSON line per input and a summary.
 *   make clean all run platform=board MODEL=resnet8
 * Runs every input with Winograd (where it pays), then with every conv direct.
 */
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram.h"
#include "bsp/ram/hyperram.h"
#include <math.h>
#include NET

static struct pi_device ram, flash, fs, cl;
static pi_fs_file_t *wf, *xf, *rf, *uf;
static uint32_t ram_base;

static void ram_rd(uint32_t off, void *dst, uint32_t n) { pi_ram_read(&ram, ram_base + off, dst, n); }
static void ram_wr(uint32_t off, const void *src, uint32_t n) { pi_ram_write(&ram, ram_base + off, (void *)src, n); }
static void file_rd(pi_fs_file_t *f, uint32_t off, void *dst, uint32_t n) {
    if (!n) return;   /* a 0-byte read corrupts L2 */
    pi_fs_seek(f, off);
    pi_fs_read(f, dst, n);
}
static void fl_rd(uint32_t off, void *dst, uint32_t n) { file_rd(wf, off, dst, n); }
static void fl_rd_ref(uint32_t off, void *dst, uint32_t n) { file_rd(rf, off, dst, n); }
static void fl_rd_u(uint32_t off, void *dst, uint32_t n) { file_rd(uf, off, dst, n); }
static uint32_t now_us(void) { return (uint32_t)pi_time_get_us(); }

static void par(void (*fn)(int, int));

#include "engine.h"

static void (*g_fn)(int, int);
static void team(void *arg) { g_fn(pi_core_id(), NC); }
static void cl_entry(void *arg) { pi_cl_team_fork(NC, team, NULL); }
static void par(void (*fn)(int, int)) {
    struct pi_cluster_task t;
    g_fn = fn;
    pi_cluster_task(&t, cl_entry, NULL);
    t.stack_size = 2048;
    t.slave_stack_size = 1536;
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
    wf = pi_fs_open(&fs, MODEL ".bin", 0); xf = pi_fs_open(&fs, MODEL ".input", 0); rf = pi_fs_open(&fs, MODEL ".ref", 0);
    uf = pi_fs_open(&fs, MODEL ".u", 0);
    if (!wf || !xf || !rf || !uf) { printf("FAIL: files\n"); pmsis_exit(-4); }
    pi_cluster_conf_init(&cconf); pi_open_from_conf(&cl, &cconf);
    if (pi_cluster_open(&cl)) { printf("FAIL: cluster\n"); pmsis_exit(-5); }
    /* the largest L2 block we can get, less some room for the drivers */
    for (A_n = 440 * 256; A_n > 16 * 256 && !(A = pi_l2_malloc(A_n * 4)); A_n -= 4 * 256);
    pi_l2_free(A, A_n * 4);
    A_n -= 8 * 256;
    A = pi_l2_malloc(A_n * 4);
    L1S_n = NC * (32 + CC) * 36;   /* blocks of up to 32 output channels */
    L1S = pi_cl_l1_malloc(&cl, L1S_n * 4);
    if (!L1S) L1S_n = 0;
    const Ten *in = &TENS[IN_ID];
    int n_in = xf->size / (numel(in) * 4), out_n = 0;
    for (int i = 0; i < N_OUTS; i++) out_n += numel(&TENS[OUTS[i]]);
    printf("{\"model\": \"%s\", \"fc_hz\": %d, \"cl_hz\": %d, \"arena_kb\": %d, \"l1_scratch_kb\": %d, \"ram_kb\": %d, \"weights_kb\": %d, \"inputs\": %d}\n",
           MODEL, pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), A_n / 256, L1S_n / 256,
           RAM_USED / 1024, wf->size / 1024, n_in);
    int any_wino = 0;
    for (int i = 0; i < N_OPS; i++) any_wino |= wino_pays(&OPS[i]);
    for (use_wino = any_wino; use_wino >= 0; use_wino--) {
        for (int img = 0; img < n_in; img++) {
            /* input file -> HyperRAM through the arena */
            for (int i0 = 0; i0 < numel(in); i0 += A_n) {
                int n = numel(in) - i0 < A_n ? numel(in) - i0 : A_n;
                file_rd(xf, ((uint32_t)img * numel(in) + i0) * 4, A, n * 4);
                ram_wr(in->off + i0 * 4, A, n * 4);
            }
            uint32_t t0 = now_us();
            run_net();
            uint32_t total = now_us() - t0;
            if (fail) pmsis_exit(-6);
            printf("{\"model\": \"%s\", \"input\": %d, \"wino\": %d, \"total_ms\": %u, \"io_ms\": %u", MODEL, img, use_wino, total / 1000, io_us / 1000);
            for (int c = 0; c < N_CAT; c++) printf(", \"%s_ms\": %u", CAT_NAME[c], cat_us[c] / 1000);
            int ref_off = img * out_n;
            for (int i = 0; i < N_OUTS; i++) {
                uint32_t h = 2166136261u; float err = 0, mx = 0;
                const Ten *t = &TENS[OUTS[i]];
                check_output(t, ref_off, &h, &err, &mx);
                ref_off += numel(t);
                printf(", \"out%d_hash\": \"%08x\", \"out%d_err_e9\": %d, \"out%d_max_e3\": %d", i, (unsigned)h, i,
                       (int)(err * 1e9f), i, (int)(mx * 1e3f));
            }
            if (N_OUTS == 1 && numel(&TENS[OUTS[0]]) <= 1000) {   /* classifier: predicted class */
                const Ten *t = &TENS[OUTS[0]]; int best = 0;
                ram_rd(t->off, A, numel(t) * 4);
                for (int i = 1; i < numel(t); i++) if (A[i] > A[best]) best = i;
                printf(", \"pred\": %d", best);
            }
            printf("}\n");
            if (img == 0) {   /* per-op times, first input of each pass */
                printf("{\"model\": \"%s\", \"wino\": %d, \"weight_transform_ms\": %u, \"op_us\": [", MODEL, use_wino, wt_us / 1000);
                for (int i = 0; i < N_OPS; i++) printf("%s%u", i ? ", " : "", op_us[i]);
                printf("]}\n");
            }
        }
    }
    printf("{\"done\": 1}\n");
    pmsis_exit(0);
}

int main(void) { return pmsis_kickoff((void *)run); }
