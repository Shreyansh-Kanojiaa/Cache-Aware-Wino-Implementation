/* TensorFlow Lite Micro (reference kernels) on the GAP8 fabric controller: MLPerf Tiny ResNet-8 int8. */
#include "pmsis.h"
#include "data.h"

int tflm_init(const void *model, unsigned char *arena, int arena_bytes);
signed char *tflm_input(int *n, float *scale, int *zp);
signed char *tflm_output(int *n);
int tflm_invoke(void);

#define ARENA_BYTES (96 * 1024)
static unsigned char arena[ARENA_BYTES] __attribute__((aligned(16)));

static void run(void) {
    int used = tflm_init(MODEL, arena, ARENA_BYTES);
    if (used < 0) { printf("FAIL: tflm_init %d\n", used); pmsis_exit(-1); }
    int n, zp, on; float scale;
    signed char *in = tflm_input(&n, &scale, &zp);
    printf("{\"engine\": \"tflm\", \"model\": \"resnet8_mlperf_int8\", \"fc_hz\": %d, \"arena_used\": %d, \"in_scale_e6\": %d, \"in_zp\": %d}\n",
           pi_freq_get(PI_FREQ_DOMAIN_FC), used, (int)(scale * 1e6f), zp);
    int correct = 0;
    for (int img = 0; img < N_IMG; img++) {
        for (int i = 0; i < n; i++) {   /* MLPerf's input: raw pixels, scale 1, zero point -128 */
            int q = (int)(IMAGES[img][i] / scale + 0.5f) + zp;
            in[i] = q < -128 ? -128 : (q > 127 ? 127 : q);
        }
        pi_perf_conf(1 << PI_PERF_CYCLES); pi_perf_reset(); pi_perf_start();
        int rc = tflm_invoke();
        pi_perf_stop();
        unsigned cyc = pi_perf_read(PI_PERF_CYCLES);
        signed char *out = tflm_output(&on);
        int best = 0; unsigned h = 2166136261u;
        for (int i = 0; i < on; i++) { if (out[i] > out[best]) best = i; h = (h ^ (unsigned char)out[i]) * 16777619u; }
        correct += best == LABELS[img];
        printf("{\"engine\": \"tflm\", \"input\": %d, \"rc\": %d, \"cycles\": %u, \"ms\": %u, \"pred\": %d, \"label\": %d, \"out_hash\": \"%08x\"}\n",
               img, rc, cyc, cyc / (pi_freq_get(PI_FREQ_DOMAIN_FC) / 1000), best, LABELS[img], h);
    }
    printf("{\"engine\": \"tflm\", \"correct\": %d, \"of\": %d}\n{\"done\": 1}\n", correct, N_IMG);
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
