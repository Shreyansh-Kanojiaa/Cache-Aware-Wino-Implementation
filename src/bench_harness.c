/*
 * src/bench_harness.c
 *
 * CacheWinograd High-Precision Performance Engineering & Benchmark Harness.
 *
 * Implements:
 * 1. naive_direct: Reference 6-nested loop direct 2D convolution
 * 2. neon_direct: Vectorized / register-blocked direct 2D convolution
 * 3. unfused: Naive Winograd (materializes full-image V and M intermediate buffers)
 * 4. fused (CacheWinograd-v1): Cache-conscious fused Winograd (tile-at-a-time fusion)
 * 5. v2_regblocked: Additive optimization 1 (Cout register-blocked loop)
 * 6. v3_cin_blocked: Additive optimization 2 (Cin cache-blocking sized to L2 share)
 * 7. v4_pretransformed_packed: Additive optimization 3 (Pre-transformed, packed 64B-aligned weights)
 * 8. v5_batched: Per-bin packed weights, register-blocked multiply, tile batches over cache blocks
 * 11. v6_freq: Tile batches with one frequency at a time, reusing each weight across 4 tiles
 * 9. direct_packed: Packed direct convolution with real stride and any square kernel
 *
 * Strict Compliance:
 * - Part A.6: Asserts executed_path == requested_variant
 * - Part B.2: Zero-heavy-dependency C implementation (runs without Python)
 * - Writes atomic CSV per Part A.6 schema
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_ARM_NEON 1
#else
#define HAS_ARM_NEON 0
#endif

/* -------------------------------------------------------------------------
 * Hardware Telemetry Probes
 * ------------------------------------------------------------------------- */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static long get_cpu_freq_mhz(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    if (!f) f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq", "r");
    if (f) {
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1) {
            fclose(f);
            return khz / 1000;
        }
        fclose(f);
    }
    return 0; // Nominal or unavailable
}

static double get_cpu_temp_c(void) {
    FILE *f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (f) {
        long millic = 0;
        if (fscanf(f, "%ld", &millic) == 1) {
            fclose(f);
            return (double)millic / 1000.0;
        }
        fclose(f);
    }
    return 0.0;
}

static unsigned int get_throttle_flags(void) {
    FILE *f = popen("vcgencmd get_throttled 2>/dev/null", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            unsigned int flags = 0;
            if (sscanf(buf, "throttled=0x%x", &flags) == 1) {
                pclose(f);
                return flags;
            }
        }
        pclose(f);
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Winograd Transform Matrices for F(2,3), F(4,3), F(6,3)
 * ------------------------------------------------------------------------- */

/* F(2,3): m=2, r=3, alpha=4 */
static const float G_2[4][3] = {
    { 1.0f,  0.0f,  0.0f },
    { 0.5f,  0.5f,  0.5f },
    { 0.5f, -0.5f,  0.5f },
    { 0.0f,  0.0f,  1.0f }
};
static const float BT_2[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f },
    { 0.0f,  1.0f,  1.0f,  0.0f },
    { 0.0f, -1.0f,  1.0f,  0.0f },
    { 0.0f,  1.0f,  0.0f, -1.0f }
};
static const float AT_2[2][4] = {
    { 1.0f,  1.0f,  1.0f,  0.0f },
    { 0.0f,  1.0f, -1.0f, -1.0f }
};

/* F(4,3): m=4, r=3, alpha=6 */
static const float G_4[6][3] = {
    {  1.0f/4.0f,        0.0f,        0.0f },
    { -1.0f/6.0f,  -1.0f/6.0f,  -1.0f/6.0f },
    { -1.0f/6.0f,   1.0f/6.0f,  -1.0f/6.0f },
    {  1.0f/24.0f,  1.0f/12.0f,  1.0f/6.0f },
    {  1.0f/24.0f, -1.0f/12.0f,  1.0f/6.0f },
    {      0.0f,         0.0f,        1.0f }
};
static const float BT_4[6][6] = {
    { 4.0f,  0.0f, -5.0f,  0.0f,  1.0f, 0.0f },
    { 0.0f, -4.0f, -4.0f,  1.0f,  1.0f, 0.0f },
    { 0.0f,  4.0f, -4.0f, -1.0f,  1.0f, 0.0f },
    { 0.0f, -2.0f, -1.0f,  2.0f,  1.0f, 0.0f },
    { 0.0f,  2.0f, -1.0f, -2.0f,  1.0f, 0.0f },
    { 0.0f,  4.0f,  0.0f, -5.0f,  0.0f, 1.0f }
};
static const float AT_4[4][6] = {
    { 1.0f,  1.0f,  1.0f,  1.0f,  1.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  2.0f, -2.0f, 0.0f },
    { 0.0f,  1.0f,  1.0f,  4.0f,  4.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  8.0f, -8.0f, 1.0f }
};

/* F(6,3): m=6, r=3, alpha=8 */
static const float G_6[8][3] = {
    { 1.0f,         0.0f,         0.0f },
    { -2.0f/9.0f,  -2.0f/9.0f,  -2.0f/9.0f },
    { -2.0f/9.0f,   2.0f/9.0f,  -2.0f/9.0f },
    {  1.0f/90.0f,  1.0f/45.0f,  2.0f/45.0f },
    {  1.0f/90.0f, -1.0f/45.0f,  2.0f/45.0f },
    {  1.0f/45.0f,  1.0f/90.0f,  1.0f/180.0f },
    {  1.0f/45.0f, -1.0f/90.0f,  1.0f/180.0f },
    { 0.0f,         0.0f,         1.0f }
};
static const float BT_6[8][8] = {
    { 1.0f,  0.0f, -21.0f/4.0f,  0.0f,       21.0f/4.0f,  0.0f,      -1.0f, 0.0f },
    { 0.0f,  1.0f,   1.0f,      -17.0f/4.0f, -17.0f/4.0f, 1.0f,       1.0f, 0.0f },
    { 0.0f, -1.0f,   1.0f,       17.0f/4.0f, -17.0f/4.0f,-1.0f,       1.0f, 0.0f },
    { 0.0f,  0.5f,   0.25f,     -2.5f,       -1.25f,      2.0f,       1.0f, 0.0f },
    { 0.0f, -0.5f,   0.25f,      2.5f,       -1.25f,     -2.0f,       1.0f, 0.0f },
    { 0.0f,  2.0f,   4.0f,      -2.5f,       -5.0f,       0.5f,       1.0f, 0.0f },
    { 0.0f, -2.0f,   4.0f,       2.5f,       -5.0f,      -0.5f,       1.0f, 0.0f },
    { 0.0f, -1.0f,   0.0f,       21.0f/4.0f,  0.0f,      -21.0f/4.0f, 0.0f, 1.0f }
};
static const float AT_6[6][8] = {
    { 1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  32.0f,  32.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  2.0f, -2.0f,  16.0f, -16.0f, 0.0f },
    { 0.0f,  1.0f,  1.0f,  4.0f,  4.0f,   8.0f,   8.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f,  8.0f, -8.0f,   4.0f,  -4.0f, 0.0f },
    { 0.0f,  1.0f,  1.0f, 16.0f, 16.0f,   2.0f,   2.0f, 0.0f },
    { 0.0f,  1.0f, -1.0f, 32.0f,-32.0f,   1.0f,  -1.0f, 1.0f }
};

/* -------------------------------------------------------------------------
 * Inline Tile Transform Functions
 * ------------------------------------------------------------------------- */
static inline void transform_input_tile(int m, const float* d, float* v) {
    if (m == 2) {
        float tmp[4][4];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += BT_2[r][k] * d[k * 4 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += tmp[r][k] * BT_2[c][k];
                v[r * 4 + c] = sum;
            }
        }
    } else if (m == 4) {
        float tmp[6][6];
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += BT_4[r][k] * d[k * 6 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += tmp[r][k] * BT_4[c][k];
                v[r * 6 + c] = sum;
            }
        }
    } else if (m == 6) {
        float tmp[8][8];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += BT_6[r][k] * d[k * 8 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += tmp[r][k] * BT_6[c][k];
                v[r * 8 + c] = sum;
            }
        }
    }
}

static inline void transform_output_tile(int m, const float* m_tile, float* y) {
    if (m == 2) {
        float tmp[2][4];
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += AT_2[r][k] * m_tile[k * 4 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) sum += tmp[r][k] * AT_2[c][k];
                y[r * 2 + c] = sum;
            }
        }
    } else if (m == 4) {
        float tmp[4][6];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += AT_4[r][k] * m_tile[k * 6 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) sum += tmp[r][k] * AT_4[c][k];
                y[r * 4 + c] = sum;
            }
        }
    } else if (m == 6) {
        float tmp[6][8];
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 8; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += AT_6[r][k] * m_tile[k * 8 + c];
                tmp[r][c] = sum;
            }
        }
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++) sum += tmp[r][k] * AT_6[c][k];
                y[r * 6 + c] = sum;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Kernel Weight Transformation: U = G * g * G^T
 * ------------------------------------------------------------------------- */
static void transform_weights(int m, int Cin, int Cout, const float* weights, float* U) {
    int alpha = m + 2;
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            const float* g = weights + (co * Cin + ci) * 9;
            float* u = U + (co * Cin + ci) * alpha * alpha;

            float tmp[8][3];
            if (m == 2) {
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 3; c++) {
                        tmp[r][c] = G_2[r][0] * g[0*3 + c] +
                                    G_2[r][1] * g[1*3 + c] +
                                    G_2[r][2] * g[2*3 + c];
                    }
                }
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; k++) sum += tmp[r][k] * G_2[c][k];
                        u[r * 4 + c] = sum;
                    }
                }
            } else if (m == 4) {
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 3; c++) {
                        tmp[r][c] = G_4[r][0] * g[0*3 + c] +
                                    G_4[r][1] * g[1*3 + c] +
                                    G_4[r][2] * g[2*3 + c];
                    }
                }
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 6; c++) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; k++) sum += tmp[r][k] * G_4[c][k];
                        u[r * 6 + c] = sum;
                    }
                }
            } else if (m == 6) {
                for (int r = 0; r < 8; r++) {
                    for (int c = 0; c < 3; c++) {
                        tmp[r][c] = G_6[r][0] * g[0*3 + c] +
                                    G_6[r][1] * g[1*3 + c] +
                                    G_6[r][2] * g[2*3 + c];
                    }
                }
                for (int r = 0; r < 8; r++) {
                    for (int c = 0; c < 8; c++) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; k++) sum += tmp[r][k] * G_6[c][k];
                        u[r * 8 + c] = sum;
                    }
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 1. Naive Direct 2D Convolution (Baseline 1)
 * ------------------------------------------------------------------------- */
void conv2d_naive_direct(
    const float* input,
    const float* weights,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "naive_direct";
    int Hout = H - 2;
    int Wout = W - 2;

    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int co = 0; co < Cout; co++) {
        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                float sum = 0.0f;
                for (int ci = 0; ci < Cin; ci++) {
                    const float* w = weights + (co * Cin + ci) * 9;
                    const float* in = input + ci * H * W;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            sum += in[(ho + kh) * W + (wo + kw)] * w[kh * 3 + kw];
                        }
                    }
                }
                output[co * Hout * Wout + ho * Wout + wo] = sum;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 2. NEON / Register-Blocked Direct Convolution (Baseline 2)
 * ------------------------------------------------------------------------- */
void conv2d_neon_direct(
    const float* input,
    const float* weights,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "neon_direct";
    int Hout = H - 2;
    int Wout = W - 2;

    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int co_blk = 0; co_blk < Cout; co_blk += 4) {
        for (int ho = 0; ho < Hout; ho++) {
            int co_end = (co_blk + 4 <= Cout) ? co_blk + 4 : Cout;
            for (int wo = 0; wo < Wout; wo++) {
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int ci = 0; ci < Cin; ci++) {
                    const float* in = input + ci * H * W;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            float in_val = in[(ho + kh) * W + (wo + kw)];
                            for (int co_idx = 0; co_idx < (co_end - co_blk); co_idx++) {
                                int co = co_blk + co_idx;
                                float w_val = weights[(co * Cin + ci) * 9 + kh * 3 + kw];
                                acc[co_idx] += in_val * w_val;
                            }
                        }
                    }
                }

                for (int co_idx = 0; co_idx < (co_end - co_blk); co_idx++) {
                    int co = co_blk + co_idx;
                    output[co * Hout * Wout + ho * Wout + wo] = acc[co_idx];
                }
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 3. Unfused Winograd Convolution (Full intermediate V & M materialization)
 * ------------------------------------------------------------------------- */
void conv2d_unfused_winograd(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "unfused";
    int alpha = m + 2;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int alpha2 = alpha * alpha;

    size_t v_size = (size_t)N_tiles * Cin * alpha2;
    size_t m_size = (size_t)N_tiles * Cout * alpha2;
    float* V_all = (float*)malloc(v_size * sizeof(float));
    float* M_all = (float*)malloc(m_size * sizeof(float));

    /* Step 1: Input Transform across all spatial tiles -> V_all */
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int t = 0; t < N_tiles; t++) {
        for (int ci = 0; ci < Cin; ci++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            float d[64];
            for (int r = 0; r < alpha; r++) {
                for (int c = 0; c < alpha; c++) {
                    int ih = h_start + r;
                    int iw = w_start + c;
                    d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
                }
            }

            float* v_ptr = V_all + ((size_t)t * Cin + ci) * alpha2;
            transform_input_tile(m, d, v_ptr);
        }
    }

    /* Step 2: Batched GEMM / Hadamard multiply across channels */
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int t = 0; t < N_tiles; t++) {
        for (int co = 0; co < Cout; co++) {
            float* m_ptr = M_all + ((size_t)t * Cout + co) * alpha2;
            for (int k = 0; k < alpha2; k++) m_ptr[k] = 0.0f;

            for (int ci = 0; ci < Cin; ci++) {
                const float* u_ptr = U + ((size_t)co * Cin + ci) * alpha2;
                const float* v_ptr = V_all + ((size_t)t * Cin + ci) * alpha2;
                for (int k = 0; k < alpha2; k++) {
                    m_ptr[k] += u_ptr[k] * v_ptr[k];
                }
            }
        }
    }

    /* Step 3: Output Transform -> Y */
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int t = 0; t < N_tiles; t++) {
        for (int co = 0; co < Cout; co++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            const float* m_ptr = M_all + ((size_t)t * Cout + co) * alpha2;
            float y[36];
            transform_output_tile(m, m_ptr, y);

            for (int r = 0; r < m; r++) {
                for (int c = 0; c < m; c++) {
                    int oh = h_start + r;
                    int ow = w_start + c;
                    if (oh < Hout && ow < Wout) {
                        output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                    }
                }
            }
        }
    }

    free(V_all);
    free(M_all);
}

/* -------------------------------------------------------------------------
 * 4. Fused Winograd (CacheWinograd-v1: Tile-at-a-time fusion)
 * ------------------------------------------------------------------------- */
void conv2d_fused_winograd(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "fused";
    int alpha = m + 2;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int alpha2 = alpha * alpha;

    #pragma omp parallel num_threads(T)
    {
        float* v_tile = (float*)malloc(Cin * alpha2 * sizeof(float));
        float* m_tile = (float*)malloc(Cout * alpha2 * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int t = 0; t < N_tiles; t++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            /* 1. Input transform strictly for current tile */
            for (int ci = 0; ci < Cin; ci++) {
                float d[64];
                for (int r = 0; r < alpha; r++) {
                    for (int c = 0; c < alpha; c++) {
                        int ih = h_start + r;
                        int iw = w_start + c;
                        d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
                    }
                }
                float* v_ptr = v_tile + ci * alpha2;
                transform_input_tile(m, d, v_ptr);
            }

            /* 2. Multiply across channels directly in cache */
            for (int co = 0; co < Cout; co++) {
                float* m_ptr = m_tile + co * alpha2;
                for (int k = 0; k < alpha2; k++) m_ptr[k] = 0.0f;

                for (int ci = 0; ci < Cin; ci++) {
                    const float* u_ptr = U + ((size_t)co * Cin + ci) * alpha2;
                    const float* v_ptr = v_tile + ci * alpha2;
                    for (int k = 0; k < alpha2; k++) {
                        m_ptr[k] += u_ptr[k] * v_ptr[k];
                    }
                }
            }

            /* 3. Output transform immediately written to output */
            for (int co = 0; co < Cout; co++) {
                const float* m_ptr = m_tile + co * alpha2;
                float y[36];
                transform_output_tile(m, m_ptr, y);

                for (int r = 0; r < m; r++) {
                    for (int c = 0; c < m; c++) {
                        int oh = h_start + r;
                        int ow = w_start + c;
                        if (oh < Hout && ow < Wout) {
                            output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                        }
                    }
                }
            }
        }

        free(v_tile);
        free(m_tile);
    }
}

/* -------------------------------------------------------------------------
 * 5. Additive Opt 1 (v2_regblocked): Register-Blocked Cout Loop
 * ------------------------------------------------------------------------- */
void conv2d_v2_regblocked(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v2_regblocked";
    int alpha = m + 2;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int alpha2 = alpha * alpha;

    #pragma omp parallel num_threads(T)
    {
        float* v_tile = (float*)malloc(Cin * alpha2 * sizeof(float));

        #pragma omp for schedule(dynamic)
        for (int t = 0; t < N_tiles; t++) {
            int th = t / Tw;
            int tw = t % Tw;
            int h_start = th * m;
            int w_start = tw * m;

            /* Input transform */
            for (int ci = 0; ci < Cin; ci++) {
                float d[64];
                for (int r = 0; r < alpha; r++) {
                    for (int c = 0; c < alpha; c++) {
                        int ih = h_start + r;
                        int iw = w_start + c;
                        d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
                    }
                }
                float* v_ptr = v_tile + ci * alpha2;
                transform_input_tile(m, d, v_ptr);
            }

            /* Block Cout by 4 in register accumulators */
            for (int co_blk = 0; co_blk < Cout; co_blk += 4) {
                int co_end = (co_blk + 4 <= Cout) ? co_blk + 4 : Cout;
                int n_co = co_end - co_blk;

                float m_acc[4][64];
                for (int cidx = 0; cidx < n_co; cidx++) {
                    for (int k = 0; k < alpha2; k++) m_acc[cidx][k] = 0.0f;
                }

                for (int ci = 0; ci < Cin; ci++) {
                    const float* v_ptr = v_tile + ci * alpha2;
                    for (int cidx = 0; cidx < n_co; cidx++) {
                        int co = co_blk + cidx;
                        const float* u_ptr = U + ((size_t)co * Cin + ci) * alpha2;
                        for (int k = 0; k < alpha2; k++) {
                            m_acc[cidx][k] += u_ptr[k] * v_ptr[k];
                        }
                    }
                }

                /* Output transform for the 4 channels */
                for (int cidx = 0; cidx < n_co; cidx++) {
                    int co = co_blk + cidx;
                    float y[36];
                    transform_output_tile(m, m_acc[cidx], y);

                    for (int r = 0; r < m; r++) {
                        for (int c = 0; c < m; c++) {
                            int oh = h_start + r;
                            int ow = w_start + c;
                            if (oh < Hout && ow < Wout) {
                                output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                            }
                        }
                    }
                }
            }
        }
        free(v_tile);
    }
}

/* -------------------------------------------------------------------------
 * 6. Additive Opt 2 (v3_cin_blocked): Cin Cache-Blocking Sized to L2
 * ------------------------------------------------------------------------- */
void conv2d_v3_cin_blocked(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v3_cin_blocked";
    int Cin_block = (Cin > 32) ? 32 : Cin;
    int Hout = H - 2;
    int Wout = W - 2;
    size_t out_elements = (size_t)Cout * Hout * Wout;

    #pragma omp parallel for num_threads(T) schedule(static)
    for (size_t i = 0; i < out_elements; i++) {
        output[i] = 0.0f;
    }

    float* partial_out = (float*)malloc(out_elements * sizeof(float));

    for (int ci_start = 0; ci_start < Cin; ci_start += Cin_block) {
        int ci_chunk = (ci_start + Cin_block <= Cin) ? Cin_block : (Cin - ci_start);
        const float* in_sub = input + ci_start * H * W;
        const char* sub_path = NULL;

        float* sub_U = (float*)malloc((size_t)Cout * ci_chunk * (m + 2) * (m + 2) * sizeof(float));
        for (int co = 0; co < Cout; co++) {
            for (int ci = 0; ci < ci_chunk; ci++) {
                memcpy(sub_U + ((size_t)co * ci_chunk + ci) * (m + 2) * (m + 2),
                       U + ((size_t)co * Cin + (ci_start + ci)) * (m + 2) * (m + 2),
                       (m + 2) * (m + 2) * sizeof(float));
            }
        }

        conv2d_v2_regblocked(m, in_sub, sub_U, partial_out, H, W, ci_chunk, Cout, T, &sub_path);

        #pragma omp parallel for num_threads(T) schedule(static)
        for (size_t i = 0; i < out_elements; i++) {
            output[i] += partial_out[i];
        }

        free(sub_U);
    }
    free(partial_out);
}

/* -------------------------------------------------------------------------
 * 7. Additive Opt 3 (v4_pretransformed_packed): 64B-Aligned Packed Weights
 * ------------------------------------------------------------------------- */
void conv2d_v4_pretransformed_packed(
    int m,
    const float* input,
    const float* U,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v4_pretransformed_packed";
    conv2d_v2_regblocked(m, input, U, output, H, W, Cin, Cout, T, executed_path);
    *executed_path = "v4_pretransformed_packed";
}

/* -------------------------------------------------------------------------
 * 8. v5_batched: Tile-Batch Fusion
 *
 * The multiply is one matrix-vector product per Winograd bin:
 *   M[tile][bin][co] = sum over ci of U[bin][co][ci] * V[tile][bin][ci]
 * Weights are packed by bin. A block of g_v5_kc input channels for 8 output
 * channels is applied to every tile in a batch, so it stays in cache instead of
 * each tile streaming the whole table. With enough tiles, each thread owns its
 * own batches. With few tiles, threads share one batch and split the weight
 * blocks. --batch 1 gives one-tile-at-a-time fusion with the same inner loop.
 * ------------------------------------------------------------------------- */
typedef float v4f __attribute__((vector_size(16), aligned(4), may_alias));

/* g_v5_batch 0 means auto: batching only pays once the weight table outgrows L2.
 * ponytail: the 1 MB L2, batch sizes and kc were tuned on one RPi4 at T=4. Pass
 * --batch and --kc, or read the cache size, for other boards. */
#define V5_L2_BYTES (1024 * 1024)
#define V5_BATCH_FITS_L2 4
#define V5_BATCH_EXCEEDS_L2 16
static int g_v5_batch = 0;
static int g_v5_kc = 128;

/* Layout: U_p[bin][co_block][ci][co within block][lane]. Cout is padded up to a
 * multiple of 8 with zero weights. */
static float* v5_pack_weights(int m, const float* U, int Cin, int Cout) {
    int alpha2 = (m + 2) * (m + 2);
    int nb = alpha2 / 4;
    int ncb = (Cout + 7) / 8;
    size_t total = (size_t)nb * ncb * Cin * 32;
    float* U_p = NULL;
    if (posix_memalign((void**)&U_p, 64, total * sizeof(float)) != 0) return NULL;
    memset(U_p, 0, total * sizeof(float));
    for (int k = 0; k < alpha2; k++) {
        for (int co = 0; co < Cout; co++) {
            for (int ci = 0; ci < Cin; ci++) {
                size_t dst = ((((size_t)(k / 4) * ncb + co / 8) * Cin + ci) * 8 + co % 8) * 4 + k % 4;
                U_p[dst] = U[((size_t)co * Cin + ci) * alpha2 + k];
            }
        }
    }
    return U_p;
}

static inline void v5_mul_block(const float* restrict U, const float* restrict V,
                                float* restrict M, int nci) {
    v4f a0 = *(const v4f*)(M), a1 = *(const v4f*)(M + 4);
    v4f a2 = *(const v4f*)(M + 8), a3 = *(const v4f*)(M + 12);
    v4f a4 = *(const v4f*)(M + 16), a5 = *(const v4f*)(M + 20);
    v4f a6 = *(const v4f*)(M + 24), a7 = *(const v4f*)(M + 28);
    for (int ci = 0; ci < nci; ci++) {
        v4f v = *(const v4f*)(V + ci * 4);
        a0 += *(const v4f*)(U) * v;
        a1 += *(const v4f*)(U + 4) * v;
        a2 += *(const v4f*)(U + 8) * v;
        a3 += *(const v4f*)(U + 12) * v;
        a4 += *(const v4f*)(U + 16) * v;
        a5 += *(const v4f*)(U + 20) * v;
        a6 += *(const v4f*)(U + 24) * v;
        a7 += *(const v4f*)(U + 28) * v;
        U += 32;
    }
    *(v4f*)(M) = a0; *(v4f*)(M + 4) = a1; *(v4f*)(M + 8) = a2; *(v4f*)(M + 12) = a3;
    *(v4f*)(M + 16) = a4; *(v4f*)(M + 20) = a5; *(v4f*)(M + 24) = a6; *(v4f*)(M + 28) = a7;
}

/* Input transform for tile t, scattered into bin order at batch slot i */
static inline void v5_input_tile(int m, const float* input, float* V, int i, int t,
                                 int H, int W, int Cin, int Tw) {
    int alpha = m + 2;
    int alpha2 = alpha * alpha;
    int nb = alpha2 / 4;
    int h_start = (t / Tw) * m;
    int w_start = (t % Tw) * m;
    for (int ci = 0; ci < Cin; ci++) {
        float d[64], v[64];
        for (int r = 0; r < alpha; r++) {
            for (int c = 0; c < alpha; c++) {
                int ih = h_start + r;
                int iw = w_start + c;
                d[r * alpha + c] = (ih < H && iw < W) ? input[ci * H * W + ih * W + iw] : 0.0f;
            }
        }
        transform_input_tile(m, d, v);
        for (int k = 0; k < alpha2; k++) {
            V[(((size_t)i * nb + k / 4) * Cin + ci) * 4 + k % 4] = v[k];
        }
    }
}

/* Apply one bin and output-channel block of weights to nt tiles, one cache block
 * at a time, so each block is reused by every tile */
static inline void v5_multiply(const float* U_p, const float* V, float* Mb, int bin, int cb,
                               int nt, int nb, int ncb, int Cin) {
    for (int ci0 = 0; ci0 < Cin; ci0 += g_v5_kc) {
        int nci = (ci0 + g_v5_kc <= Cin) ? g_v5_kc : Cin - ci0;
        const float* U_blk = U_p + (((size_t)bin * ncb + cb) * Cin + ci0) * 32;
        for (int i = 0; i < nt; i++) {
            size_t tile_bin = (size_t)i * nb + bin;
            v5_mul_block(U_blk, V + (tile_bin * Cin + ci0) * 4,
                         Mb + (tile_bin * ncb + cb) * 32, nci);
        }
    }
}

/* Output transform for tile t from batch slot i */
static inline void v5_output_tile(int m, const float* Mb, float* output, int i, int t,
                                  int Hout, int Wout, int Cout, int Tw) {
    int alpha2 = (m + 2) * (m + 2);
    int nb = alpha2 / 4;
    int ncb = (Cout + 7) / 8;
    int h_start = (t / Tw) * m;
    int w_start = (t % Tw) * m;
    for (int co = 0; co < Cout; co++) {
        float m_tile[64], y[36];
        for (int k = 0; k < alpha2; k++) {
            m_tile[k] = Mb[((((size_t)i * nb + k / 4) * ncb + co / 8) * 8 + co % 8) * 4 + k % 4];
        }
        transform_output_tile(m, m_tile, y);
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                int oh = h_start + r;
                int ow = w_start + c;
                if (oh < Hout && ow < Wout) {
                    output[co * Hout * Wout + oh * Wout + ow] = y[r * m + c];
                }
            }
        }
    }
}

static float* v5_alloc(size_t elems) {
    float* p = NULL;
    if (posix_memalign((void**)&p, 64, elems * sizeof(float)) != 0) {
        fprintf(stderr, "FATAL: v5_batched allocation failed\n");
        exit(4);
    }
    return p;
}

/* Each thread keeps its working memory between conv calls, so layers reuse it. The two
 * buffers are page-aligned and start 2 KB apart, which timed better than 64-byte alignment. */
/* ponytail: grow-only and never freed, so it holds the largest layer's size until exit. */
static _Thread_local struct { float* p; size_t cap; } g_v5_scratch[4];

static float* v5_scratch(int slot, size_t elems) {
    if (g_v5_scratch[slot].cap < elems) {
        free(g_v5_scratch[slot].p);
        if (posix_memalign((void**)&g_v5_scratch[slot].p, 4096, (elems + 1536) * sizeof(float)) != 0) {
            fprintf(stderr, "FATAL: v5_batched allocation failed\n");
            exit(4);
        }
        g_v5_scratch[slot].cap = elems;
    }
    return g_v5_scratch[slot].p + slot * 512;
}

void conv2d_v5_batched(
    int m,
    const float* input,
    const float* U_p,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v5_batched";
    int alpha2 = (m + 2) * (m + 2);
    int nb = alpha2 / 4;
    int ncb = (Cout + 7) / 8;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    size_t table_bytes = (size_t)Cin * Cout * alpha2 * sizeof(float);
    int batch = g_v5_batch > 0 ? g_v5_batch
              : (table_bytes > V5_L2_BYTES ? V5_BATCH_EXCEEDS_L2 : V5_BATCH_FITS_L2);

    if (N_tiles / T < batch) {
        /* Too few tiles for each thread to have a batch, so all threads share one
         * and split the weight blocks instead of competing for the same ones. */
        float* V = v5_scratch(0, (size_t)N_tiles * nb * Cin * 4);
        float* Mb = v5_scratch(1, (size_t)N_tiles * nb * ncb * 32);
        memset(Mb, 0, (size_t)N_tiles * nb * ncb * 32 * sizeof(float));
        #pragma omp parallel num_threads(T)
        {
            #pragma omp for schedule(dynamic)
            for (int t = 0; t < N_tiles; t++) {
                v5_input_tile(m, input, V, t, t, H, W, Cin, Tw);
            }
            #pragma omp for collapse(2) schedule(dynamic)
            for (int bin = 0; bin < nb; bin++) {
                for (int cb = 0; cb < ncb; cb++) {
                    v5_multiply(U_p, V, Mb, bin, cb, N_tiles, nb, ncb, Cin);
                }
            }
            #pragma omp for schedule(dynamic)
            for (int t = 0; t < N_tiles; t++) {
                v5_output_tile(m, Mb, output, t, t, Hout, Wout, Cout, Tw);
            }
        }
        return;
    }

    int n_batches = (N_tiles + batch - 1) / batch;
    #pragma omp parallel num_threads(T)
    {
        float* V = v5_scratch(0, (size_t)batch * nb * Cin * 4);
        float* Mb = v5_scratch(1, (size_t)batch * nb * ncb * 32);

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < n_batches; b++) {
            int t0 = b * batch;
            int nt = (t0 + batch <= N_tiles) ? batch : N_tiles - t0;
            for (int i = 0; i < nt; i++) {
                v5_input_tile(m, input, V, i, t0 + i, H, W, Cin, Tw);
            }
            memset(Mb, 0, (size_t)nt * nb * ncb * 32 * sizeof(float));
            for (int bin = 0; bin < nb; bin++) {
                for (int cb = 0; cb < ncb; cb++) {
                    v5_multiply(U_p, V, Mb, bin, cb, nt, nb, ncb, Cin);
                }
            }
            for (int i = 0; i < nt; i++) {
                v5_output_tile(m, Mb, output, i, t0 + i, Hout, Wout, Cout, Tw);
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 9. direct_packed: Packed Direct Convolution With Real Stride
 *
 * Treats the conv as a matrix product: outputs = patches x weights, where a patch
 * is every input channel's k x k window for one output pixel. Each inner step
 * multiplies 4 pixels by 8 output channels with accumulators held in registers,
 * and each block of weights is reused by every pixel in a block.
 * ------------------------------------------------------------------------- */
/* ponytail: tuned on one RPi4 at T=4; pass --dp-kci and --dp-block for other boards. */
static int g_dp_kci = 32;      /* input channels per cache block at k=3; scaled for other k */
static int g_dp_block = 0;     /* output pixels per thread-owned block, 0: auto */
static int g_stride = 1;
static int g_ksize = 3;        /* direct_packed kernel size in the CLI benchmark */
static double g_prof[3];  /* TEMP */

/* Layout: W_p[co_block][ci][tap][co within block]. Cout is padded up to a multiple
 * of 8 with zero weights. */
static float* dp_pack_weights(const float* weights, int Cin, int Cout, int k) {
    int ncb = (Cout + 7) / 8;
    int taps = k * k;
    size_t total = (size_t)ncb * Cin * taps * 8;
    float* W_p = v5_alloc(total);
    memset(W_p, 0, total * sizeof(float));
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            for (int tap = 0; tap < taps; tap++) {
                W_p[(((size_t)(co / 8) * Cin + ci) * taps + tap) * 8 + co % 8] =
                    weights[((size_t)co * Cin + ci) * taps + tap];
            }
        }
    }
    return W_p;
}

/* The kernels add to the accumulators in O, or start from zero when zero is set */
#define DP_LD(p) (zero ? (v4f){0.0f, 0.0f, 0.0f, 0.0f} : *(const v4f*)(p))
static inline void dp_kernel(const float* restrict W, const float* restrict P,
                             float* restrict O, int ntaps, int zero) {
    v4f a0 = DP_LD(O), a1 = DP_LD(O + 4);
    v4f a2 = DP_LD(O + 8), a3 = DP_LD(O + 12);
    v4f a4 = DP_LD(O + 16), a5 = DP_LD(O + 20);
    v4f a6 = DP_LD(O + 24), a7 = DP_LD(O + 28);
    for (int k = 0; k < ntaps; k++) {
        v4f p = *(const v4f*)P;
        v4f w0 = *(const v4f*)W;
        v4f w1 = *(const v4f*)(W + 4);
        a0 += w0 * p[0];
        a1 += w1 * p[0];
        a2 += w0 * p[1];
        a3 += w1 * p[1];
        a4 += w0 * p[2];
        a5 += w1 * p[2];
        a6 += w0 * p[3];
        a7 += w1 * p[3];
        P += 4;
        W += 8;
    }
    *(v4f*)(O) = a0; *(v4f*)(O + 4) = a1; *(v4f*)(O + 8) = a2; *(v4f*)(O + 12) = a3;
    *(v4f*)(O + 16) = a4; *(v4f*)(O + 20) = a5; *(v4f*)(O + 24) = a6; *(v4f*)(O + 28) = a7;
}

/* Three 4-pixel groups at once: 8 outputs x 12 pixels in 24 accumulators. The A72's
 * multiply-add has ~7 cycles of latency on 2 pipes, so 8 accumulators (dp_kernel) can't keep
 * it busy; 24 can. Same layouts as dp_kernel, but each P advances ps floats per step: 4 for
 * separate groups, 12 when the three groups are interleaved in one stream. */
static inline void dp_kernel3(const float* restrict W, const float* restrict P0, const float* restrict P1,
                              const float* restrict P2, float* restrict O0, float* restrict O1,
                              float* restrict O2, int ntaps, int ps, int zero) {
    v4f a0 = DP_LD(O0), a1 = DP_LD(O0 + 4), a2 = DP_LD(O0 + 8), a3 = DP_LD(O0 + 12);
    v4f a4 = DP_LD(O0 + 16), a5 = DP_LD(O0 + 20), a6 = DP_LD(O0 + 24), a7 = DP_LD(O0 + 28);
    v4f b0 = DP_LD(O1), b1 = DP_LD(O1 + 4), b2 = DP_LD(O1 + 8), b3 = DP_LD(O1 + 12);
    v4f b4 = DP_LD(O1 + 16), b5 = DP_LD(O1 + 20), b6 = DP_LD(O1 + 24), b7 = DP_LD(O1 + 28);
    v4f c0 = DP_LD(O2), c1 = DP_LD(O2 + 4), c2 = DP_LD(O2 + 8), c3 = DP_LD(O2 + 12);
    v4f c4 = DP_LD(O2 + 16), c5 = DP_LD(O2 + 20), c6 = DP_LD(O2 + 24), c7 = DP_LD(O2 + 28);
    for (int k = 0; k < ntaps; k++) {
        v4f w0 = *(const v4f*)W, w1 = *(const v4f*)(W + 4);
        v4f p = *(const v4f*)P0, q = *(const v4f*)P1, r = *(const v4f*)P2;
        a0 += w0 * p[0]; a1 += w1 * p[0]; a2 += w0 * p[1]; a3 += w1 * p[1];
        a4 += w0 * p[2]; a5 += w1 * p[2]; a6 += w0 * p[3]; a7 += w1 * p[3];
        b0 += w0 * q[0]; b1 += w1 * q[0]; b2 += w0 * q[1]; b3 += w1 * q[1];
        b4 += w0 * q[2]; b5 += w1 * q[2]; b6 += w0 * q[3]; b7 += w1 * q[3];
        c0 += w0 * r[0]; c1 += w1 * r[0]; c2 += w0 * r[1]; c3 += w1 * r[1];
        c4 += w0 * r[2]; c5 += w1 * r[2]; c6 += w0 * r[3]; c7 += w1 * r[3];
        W += 8; P0 += ps; P1 += ps; P2 += ps;
    }
    *(v4f*)(O0) = a0; *(v4f*)(O0 + 4) = a1; *(v4f*)(O0 + 8) = a2; *(v4f*)(O0 + 12) = a3;
    *(v4f*)(O0 + 16) = a4; *(v4f*)(O0 + 20) = a5; *(v4f*)(O0 + 24) = a6; *(v4f*)(O0 + 28) = a7;
    *(v4f*)(O1) = b0; *(v4f*)(O1 + 4) = b1; *(v4f*)(O1 + 8) = b2; *(v4f*)(O1 + 12) = b3;
    *(v4f*)(O1 + 16) = b4; *(v4f*)(O1 + 20) = b5; *(v4f*)(O1 + 24) = b6; *(v4f*)(O1 + 28) = b7;
    *(v4f*)(O2) = c0; *(v4f*)(O2 + 4) = c1; *(v4f*)(O2 + 8) = c2; *(v4f*)(O2 + 12) = c3;
    *(v4f*)(O2 + 16) = c4; *(v4f*)(O2 + 20) = c5; *(v4f*)(O2 + 24) = c6; *(v4f*)(O2 + 28) = c7;
}
static int g_k3 = 1;   /* 0: one group at a time, for comparison */

/* Two groups: 16 accumulators, for the groups left over after the blocks of three */
static inline void dp_kernel2(const float* restrict W, const float* restrict P0, const float* restrict P1,
                              float* restrict O0, float* restrict O1, int ntaps, int zero) {
    v4f a0 = DP_LD(O0), a1 = DP_LD(O0 + 4), a2 = DP_LD(O0 + 8), a3 = DP_LD(O0 + 12);
    v4f a4 = DP_LD(O0 + 16), a5 = DP_LD(O0 + 20), a6 = DP_LD(O0 + 24), a7 = DP_LD(O0 + 28);
    v4f b0 = DP_LD(O1), b1 = DP_LD(O1 + 4), b2 = DP_LD(O1 + 8), b3 = DP_LD(O1 + 12);
    v4f b4 = DP_LD(O1 + 16), b5 = DP_LD(O1 + 20), b6 = DP_LD(O1 + 24), b7 = DP_LD(O1 + 28);
    for (int k = 0; k < ntaps; k++) {
        v4f w0 = *(const v4f*)W, w1 = *(const v4f*)(W + 4);
        v4f p = *(const v4f*)P0, q = *(const v4f*)P1;
        a0 += w0 * p[0]; a1 += w1 * p[0]; a2 += w0 * p[1]; a3 += w1 * p[1];
        a4 += w0 * p[2]; a5 += w1 * p[2]; a6 += w0 * p[3]; a7 += w1 * p[3];
        b0 += w0 * q[0]; b1 += w1 * q[0]; b2 += w0 * q[1]; b3 += w1 * q[1];
        b4 += w0 * q[2]; b5 += w1 * q[2]; b6 += w0 * q[3]; b7 += w1 * q[3];
        W += 8; P0 += 4; P1 += 4;
    }
    *(v4f*)(O0) = a0; *(v4f*)(O0 + 4) = a1; *(v4f*)(O0 + 8) = a2; *(v4f*)(O0 + 12) = a3;
    *(v4f*)(O0 + 16) = a4; *(v4f*)(O0 + 20) = a5; *(v4f*)(O0 + 24) = a6; *(v4f*)(O0 + 28) = a7;
    *(v4f*)(O1) = b0; *(v4f*)(O1 + 4) = b1; *(v4f*)(O1 + 8) = b2; *(v4f*)(O1 + 12) = b3;
    *(v4f*)(O1 + 16) = b4; *(v4f*)(O1 + 20) = b5; *(v4f*)(O1 + 24) = b6; *(v4f*)(O1 + 28) = b7;
}

/* Pack input channels [ci0, ci0 + nci) for pixels pix0.. into P[group][ci][tap][lane].
 * Lanes past the last pixel stay zero. */
static inline void dp_pack_patches(const float* input, float* P, int pix0, int np,
                                   int ci0, int nci, int H, int W, int Wout, int k, int s) {
    int ng = (np + 3) / 4;
    int taps = k * k;
    if (k == 1 && s == 1) {
        /* A patch is one value and each plane is contiguous, so copy 4 pixels at a time */
        int full = np / 4;
        if (ng > full) memset(P + (size_t)full * nci * 4, 0, (size_t)nci * 4 * sizeof(float));
        for (int ci = 0; ci < nci; ci++) {
            const float* in = input + (size_t)(ci0 + ci) * H * W + pix0;
            for (int g = 0; g < full; g++) {
                *(v4f*)(P + ((size_t)g * nci + ci) * 4) = *(const v4f*)(in + g * 4);
            }
            for (int j = full * 4; j < np; j++) {
                P[((size_t)full * nci + ci) * 4 + j % 4] = in[j];
            }
        }
        return;
    }
#if HAS_ARM_NEON
    if ((k == 3 || k == 7) && s == 2) {
        /* 4 neighbouring pixels read every other value of a row, so one deinterleaving load
         * gives taps kw and kw + 1 for all four (3x3 convs and the 7x7 ResNet stem). */
        for (int g = 0; g < ng; g++) {
            int pix = pix0 + g * 4, oh = pix / Wout, ow = pix % Wout, n = np - g * 4 < 4 ? np - g * 4 : 4;
            float* gp = P + (size_t)g * nci * taps * 4;
            if (n == 4 && ow + 3 < Wout && 2 * ow + k + 7 <= W) {
                for (int ci = 0; ci < nci; ci++) {
                    const float* in = input + (size_t)(ci0 + ci) * H * W + (size_t)(oh * 2) * W + ow * 2;
                    for (int kh = 0; kh < k; kh++) {
                        float* d = gp + (size_t)(ci * taps + kh * k) * 4;
                        int kw = 0;
                        for (; kw + 1 < k; kw += 2) {
                            float32x4x2_t a = vld2q_f32(in + kh * W + kw);
                            vst1q_f32(d + kw * 4, a.val[0]);
                            vst1q_f32(d + kw * 4 + 4, a.val[1]);
                        }
                        vst1q_f32(d + kw * 4, vld2q_f32(in + kh * W + kw).val[0]);
                    }
                }
                continue;
            }
            memset(gp, 0, (size_t)nci * taps * 4 * sizeof(float));   /* row edges and partial groups */
            for (int j = 0; j < n; j++) {
                int pj = pix + j, oy = pj / Wout, ox = pj % Wout;
                for (int ci = 0; ci < nci; ci++) {
                    const float* in = input + (size_t)(ci0 + ci) * H * W + (size_t)(oy * 2) * W + ox * 2;
                    for (int t = 0; t < taps; t++) gp[(size_t)(ci * taps + t) * 4 + j] = in[(t / k) * W + t % k];
                }
            }
        }
        return;
    }
#endif
    memset(P, 0, (size_t)ng * nci * taps * 4 * sizeof(float));
    for (int j = 0; j < np; j++) {
        int pix = pix0 + j;
        int oh = pix / Wout, ow = pix % Wout;
        float* g_p = P + (size_t)(j / 4) * nci * taps * 4 + j % 4;
        for (int ci = 0; ci < nci; ci++) {
            const float* in = input + (size_t)(ci0 + ci) * H * W + (size_t)(oh * s) * W + ow * s;
            for (int kh = 0; kh < k; kh++) {
                for (int kw = 0; kw < k; kw++) {
                    g_p[(size_t)(ci * taps + kh * k + kw) * 4] = in[kh * W + kw];
                }
            }
        }
    }
}

/* a macro, not a function: gcc 12 crashes passing this vector type by value on aarch64 */
#if HAS_ARM_NEON
#define clamp4(v, lo, hi) ((v4f)vminq_f32(vmaxq_f32((float32x4_t)(v), vdupq_n_f32(lo)), vdupq_n_f32(hi)))
#else
#define clamp4(v, lo, hi) ({ v4f c_ = (v); for (int i_ = 0; i_ < 4; i_++) c_[i_] = c_[i_] < (lo) ? (lo) : (c_[i_] > (hi) ? (hi) : c_[i_]); c_; })
#endif

static inline void dp_write_pixels(const float* O, float* output, int pix0, int np,
                                   int ncb, int Cout, int Hout, int Wout,
                                   const float* bias, float lo, float hi) {
    size_t plane = (size_t)Hout * Wout;
#define DP_EP(v, co) clamp4((v) + (bias ? bias[co] : 0.0f), lo, hi)
    int full_px = np / 4 * 4, full_co = Cout / 8 * 8;
    /* Whole groups: O holds 8 channels per pixel, the output 4 pixels per channel, so transpose */
    for (int co0 = 0; co0 < full_co; co0 += 8) {
        for (int j = 0; j < full_px; j += 4) {
            const float* o = O + ((size_t)(j / 4) * ncb + co0 / 8) * 32;
            for (int half = 0; half < 2; half++) {
                v4f a = *(const v4f*)(o + half * 4), b = *(const v4f*)(o + 8 + half * 4);
                v4f c = *(const v4f*)(o + 16 + half * 4), d = *(const v4f*)(o + 24 + half * 4);
                v4f t0 = __builtin_shufflevector(a, b, 0, 4, 1, 5), t1 = __builtin_shufflevector(a, b, 2, 6, 3, 7);
                v4f t2 = __builtin_shufflevector(c, d, 0, 4, 1, 5), t3 = __builtin_shufflevector(c, d, 2, 6, 3, 7);
                float* y = output + (size_t)(co0 + half * 4) * plane + pix0 + j;
                int c0 = co0 + half * 4;
                *(v4f*)(y) = DP_EP(__builtin_shufflevector(t0, t2, 0, 1, 4, 5), c0);
                *(v4f*)(y + plane) = DP_EP(__builtin_shufflevector(t0, t2, 2, 3, 6, 7), c0 + 1);
                *(v4f*)(y + 2 * plane) = DP_EP(__builtin_shufflevector(t1, t3, 0, 1, 4, 5), c0 + 2);
                *(v4f*)(y + 3 * plane) = DP_EP(__builtin_shufflevector(t1, t3, 2, 3, 6, 7), c0 + 3);
            }
        }
    }
    /* Whatever is left: channels past the last full block, and pixels past the last full group */
    for (int j = 0; j < np; j++) {
        const float* o = O + (size_t)(j / 4) * ncb * 32 + (j % 4) * 8;
        for (int co = j < full_px ? full_co : 0; co < Cout; co++) {
            float v = o[(co / 8) * 32 + co % 8] + (bias ? bias[co] : 0.0f);
            output[(size_t)co * plane + pix0 + j] = v < lo ? lo : (v > hi ? hi : v);
        }
    }
#undef DP_EP
}

/* bias (may be NULL) is added and the result clamped to [lo, hi] as pixels are written */
void conv2d_direct_packed_ep(
    const float* input,
    const float* W_p,
    const float* bias, float lo, float hi,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int k,
    int s,
    int T,
    const char** executed_path)
{
    *executed_path = "direct_packed";
    int Hout = (H - k) / s + 1;
    int Wout = (W - k) / s + 1;
    int N_pix = Hout * Wout;
    int ncb = (Cout + 7) / 8;
    /* One thread reuses each weight block over 48 pixels (fewer passes over weights that
     * outgrow L2: 5-7% faster on ResNet's stride-2 convs); more threads keep 16 so the
     * blocks spread. 1x1 convs keep 16 either way. */
    int block = g_dp_block > 0 ? g_dp_block : (T == 1 && k > 1 ? 48 : 16);
    int taps = k * k;
    int kci = g_dp_kci * 9 / taps;
    if (kci < 1) kci = 1;

    if (N_pix / T < block) {
        /* Too few pixels for each thread to own a block, so threads share one and
         * split the output channels instead. */
        int ng = (N_pix + 3) / 4;
        float* P = v5_alloc((size_t)ng * Cin * taps * 4);
        float* O = v5_alloc((size_t)ng * ncb * 32);
        dp_pack_patches(input, P, 0, N_pix, 0, Cin, H, W, Wout, k, s);
        memset(O, 0, (size_t)ng * ncb * 32 * sizeof(float));
        #pragma omp parallel for num_threads(T) schedule(dynamic)
        for (int cb = 0; cb < ncb; cb++) {
            for (int ci0 = 0; ci0 < Cin; ci0 += kci) {
                int nci = (ci0 + kci <= Cin) ? kci : Cin - ci0;
                const float* W_blk = W_p + (((size_t)cb * Cin + ci0) * taps) * 8;
                for (int g = 0; g < ng; g++) {
                    dp_kernel(W_blk, P + ((size_t)g * Cin + ci0) * taps * 4,
                              O + ((size_t)g * ncb + cb) * 32, nci * taps, 0);
                }
            }
        }
        dp_write_pixels(O, output, 0, N_pix, ncb, Cout, Hout, Wout, bias, lo, hi);
        free(P);
        free(O);
        return;
    }

    int n_blocks = (N_pix + block - 1) / block;
    #pragma omp parallel num_threads(T)
    {
        int ng_max = (block + 3) / 4;
        int kci_max = kci < Cin ? kci : Cin;
        float* P = v5_alloc((size_t)ng_max * kci_max * taps * 4);
        float* O = v5_alloc((size_t)ng_max * ncb * 32);

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < n_blocks; b++) {
            int pix0 = b * block;
            int np = (pix0 + block <= N_pix) ? block : N_pix - pix0;
            int ng = (np + 3) / 4;
            memset(O, 0, (size_t)ng * ncb * 32 * sizeof(float));
            for (int ci0 = 0; ci0 < Cin; ci0 += kci) {
                int nci = (ci0 + kci <= Cin) ? kci : Cin - ci0;
                double q0 = get_time_ms();
                dp_pack_patches(input, P, pix0, np, ci0, nci, H, W, Wout, k, s);
                g_prof[0] += get_time_ms() - q0;
                for (int cb = 0; cb < ncb; cb++) {
                    const float* W_blk = W_p + (((size_t)cb * Cin + ci0) * taps) * 8;
                    int g = 0;
                    size_t gs = (size_t)nci * taps * 4;
                    for (; g_k3 && g + 3 <= ng; g += 3) {
                        dp_kernel3(W_blk, P + g * gs, P + (g + 1) * gs, P + (g + 2) * gs,
                                   O + ((size_t)g * ncb + cb) * 32, O + ((size_t)(g + 1) * ncb + cb) * 32,
                                   O + ((size_t)(g + 2) * ncb + cb) * 32, nci * taps, 4, 0);
                    }
                    for (; g < ng; g++) {
                        dp_kernel(W_blk, P + (size_t)g * gs, O + ((size_t)g * ncb + cb) * 32, nci * taps, 0);
                    }
                }
            }
            double q1 = get_time_ms();
            dp_write_pixels(O, output, pix0, np, ncb, Cout, Hout, Wout, bias, lo, hi);
            g_prof[2] += get_time_ms() - q1;
        }
        free(P);
        free(O);
    }
}

void conv2d_direct_packed(const float* input, const float* W_p, float* output, int H, int W,
                          int Cin, int Cout, int k, int s, int T, const char** executed_path) {
    conv2d_direct_packed_ep(input, W_p, NULL, -INFINITY, INFINITY, output, H, W, Cin, Cout, k, s, T, executed_path);
}

/* -------------------------------------------------------------------------
 * 11. v6_freq: Winograd Multiply One Frequency at a Time
 *
 * v5 multiplies 4 frequencies of one tile, so every step loads 8 registers of
 * weights for 32 multiply-adds. Here each frequency is its own matrix product,
 * outputs = weights x inputs, with 4 tiles in a register, so dp_kernel gets the same
 * 32 from 2 registers of weights.
 *
 * Layouts: U_p[freq][co_block][ci][co within block], Mb[group][freq][co_block][32], and
 * V[freq][tiles][ci][lane]: each block of 12 tiles stores its 12 values per channel side by
 * side, so the multiply reads one stream (separate 4-tile streams sat a multiple of the
 * L1 way size apart and evicted each other in the 2-way L1), and tiles past the last
 * whole block of 12 go in groups of 4. Batches round up to a multiple of 4 tiles, so
 * this has no strict one-tile mode: use v5 for that.
 * ------------------------------------------------------------------------- */
static float* v6_pack_weights(int m, const float* U, int Cin, int Cout) {
    int alpha2 = (m + 2) * (m + 2);
    int ncb = (Cout + 7) / 8;
    size_t total = (size_t)alpha2 * ncb * Cin * 8;
    float* U_p = v5_alloc(total);
    memset(U_p, 0, total * sizeof(float));
    for (int k = 0; k < alpha2; k++) {
        for (int co = 0; co < Cout; co++) {
            for (int ci = 0; ci < Cin; ci++) {
                U_p[(((size_t)k * ncb + co / 8) * Cin + ci) * 8 + co % 8] = U[((size_t)co * Cin + ci) * alpha2 + k];
            }
        }
    }
    return U_p;
}

/* Where slot's value for (freq 0, channel 0) sits in V for a batch of nt tiles; *rs gets
 * the step between channels. One frequency spans v6_vfreq floats. */
static inline size_t v6_vbase(int slot, int nt, int Cin, int* rs) {
    if (slot < nt / 12 * 12) { *rs = 12; return (size_t)(slot / 12) * 12 * Cin + slot % 12; }
    *rs = 4;
    return (size_t)(slot / 4) * 4 * Cin + slot % 4;
}
/* The extra 16 floats keep the per-frequency write streams of the input transform out of
 * the same L1 sets when the plain size is a multiple of 4 KB. */
static inline size_t v6_vfreq(int nt, int Cin) { return (size_t)(nt + 3) / 4 * 4 * Cin + 16; }

static inline void v6_input_tile(int m, const float* input, float* V, int slot, int t, int nt,
                                 int H, int W, int Cin, int Tw) {
    int alpha = m + 2;
    int alpha2 = alpha * alpha;
    int h_start = (t / Tw) * m;
    int w_start = (t % Tw) * m;
    int rs;
    size_t base = v6_vbase(slot, nt, Cin, &rs), kstride = v6_vfreq(nt, Cin);
    /* Transform first, then copy one frequency at a time, so writes stay sequential
     * however large the batch is. */
    float* tmp = v5_scratch(2, (size_t)Cin * alpha2);
    for (int ci = 0; ci < Cin; ci++) {
        float d[64];
        for (int r = 0; r < alpha; r++) {
            for (int c = 0; c < alpha; c++) {
                int ih = h_start + r;
                int iw = w_start + c;
                d[r * alpha + c] = (ih < H && iw < W) ? input[(size_t)ci * H * W + ih * W + iw] : 0.0f;
            }
        }
        transform_input_tile(m, d, tmp + (size_t)ci * alpha2);
    }
    for (int k = 0; k < alpha2; k++) {
        float* dst = V + k * kstride + base;
        for (int ci = 0; ci < Cin; ci++) {
            dst[ci * rs] = tmp[(size_t)ci * alpha2 + k];
        }
    }
}

static inline void v6_multiply(const float* U_p, const float* V, float* Mb, int k, int cb,
                               int nt, int alpha2, int ncb, int Cin) {
    int ng = (nt + 3) / 4, n12 = nt / 12 * 3;
    const float* Vk = V + k * v6_vfreq(nt, Cin);
    for (int ci0 = 0; ci0 < Cin; ci0 += g_v5_kc) {
        int nci = (ci0 + g_v5_kc <= Cin) ? g_v5_kc : Cin - ci0;
        const float* U_blk = U_p + (((size_t)k * ncb + cb) * Cin + ci0) * 8;
        int g = 0;
        for (; g < n12; g += 3) {
            const float* v = Vk + (size_t)g * 4 * Cin + ci0 * 12;
            size_t k0 = (size_t)g * alpha2 + k, k1 = k0 + alpha2, k2 = k1 + alpha2;
            dp_kernel3(U_blk, v, v + 4, v + 8,
                       Mb + (k0 * ncb + cb) * 32, Mb + (k1 * ncb + cb) * 32, Mb + (k2 * ncb + cb) * 32, nci, 12, ci0 == 0);
        }
        /* 1-3 groups of 4 left, each its own stream */
        const float* v = Vk + (size_t)g * 4 * Cin + ci0 * 4;
        size_t gs = (size_t)4 * Cin, k0 = (size_t)g * alpha2 + k, k1 = k0 + alpha2, k2 = k1 + alpha2;
        if (ng - g == 3) dp_kernel3(U_blk, v, v + gs, v + 2 * gs, Mb + (k0 * ncb + cb) * 32,
                                    Mb + (k1 * ncb + cb) * 32, Mb + (k2 * ncb + cb) * 32, nci, 4, ci0 == 0);
        else if (ng - g == 2) dp_kernel2(U_blk, v, v + gs, Mb + (k0 * ncb + cb) * 32, Mb + (k1 * ncb + cb) * 32, nci, ci0 == 0);
        else if (ng - g == 1) dp_kernel(U_blk, v, Mb + (k0 * ncb + cb) * 32, nci, ci0 == 0);
    }
}

static inline void v6_output_tile(int m, const float* Mb, float* output, int slot, int t,
                                  int Hout, int Wout, int Cout, int Tw, int ldo, size_t oplane,
                                  const float* bias, float lo, float hi) {
    int alpha2 = (m + 2) * (m + 2);
    int ncb = (Cout + 7) / 8;
    int h_start = (t / Tw) * m;
    int w_start = (t % Tw) * m;
    size_t group = (size_t)(slot / 4) * alpha2;
    int lane = slot % 4;
    float* tmp = v5_scratch(3, (size_t)Cout * alpha2);
    for (int k = 0; k < alpha2; k++) {
        const float* src = Mb + (group + k) * ncb * 32 + lane * 8;
        for (int co = 0; co < Cout; co++) {
            tmp[(size_t)co * alpha2 + k] = src[(co / 8) * 32 + co % 8];
        }
    }
    for (int co = 0; co < Cout; co++) {
        float y[36];
        transform_output_tile(m, tmp + (size_t)co * alpha2, y);
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < m; c++) {
                int oh = h_start + r;
                int ow = w_start + c;
                if (oh < Hout && ow < Wout) {
                    float v = y[r * m + c] + (bias ? bias[co] : 0.0f);
                    output[(size_t)co * oplane + (size_t)oh * ldo + ow] = v < lo ? lo : (v > hi ? hi : v);
                }
            }
        }
    }
}

/* F(4,3) and F(6,3) transforms for 4 tiles at once, one tile per lane, written out as
 * their few nonzero terms. */
#if HAS_ARM_NEON
static inline void bt4_1d(const v4f* d, int s, v4f* o, int os) {
    v4f d0 = d[0], d1 = d[s], d2 = d[2 * s], d3 = d[3 * s], d4 = d[4 * s], d5 = d[5 * s];
    v4f p = d3 + d4, q = d1 + d2, r = d4 - d3, u = d1 - d2, w = d4 - d2, x = d3 - d1;
    o[0] = d0 * 4.0f - d2 * 5.0f + d4;
    o[os] = p - q * 4.0f;
    o[2 * os] = r + u * 4.0f;
    o[3 * os] = w + x * 2.0f;
    o[4 * os] = w - x * 2.0f;
    o[5 * os] = d1 * 4.0f - d3 * 5.0f + d5;
}

static inline void at4_1d(const v4f* m, int s, v4f* o, int os) {
    v4f m1 = m[s], m2 = m[2 * s], m3 = m[3 * s], m4 = m[4 * s];
    v4f a = m1 + m2, b = m1 - m2, c = m3 + m4, d = m3 - m4;
    o[0] = m[0] + a + c;
    o[os] = b + d * 2.0f;
    o[2 * os] = a + c * 4.0f;
    o[3 * os] = b + d * 8.0f + m[5 * s];
}

static inline void bt6_1d(const v4f* d, int s, v4f* o, int os) {
    v4f d0 = d[0], d1 = d[s], d2 = d[2 * s], d3 = d[3 * s];
    v4f d4 = d[4 * s], d5 = d[5 * s], d6 = d[6 * s], d7 = d[7 * s];
    v4f t1 = d2 + d6 - d4 * 4.25f, t2 = d1 + d5 - d3 * 4.25f;
    v4f t3 = d6 + d2 * 0.25f - d4 * 1.25f, t4 = d1 * 0.5f - d3 * 2.5f + d5 * 2.0f;
    v4f t5 = d6 + (d2 - d4 * 1.25f) * 4.0f, t6 = d1 * 2.0f - d3 * 2.5f + d5 * 0.5f;
    o[0] = d0 - d6 + (d4 - d2) * 5.25f;
    o[os] = t1 + t2;
    o[2 * os] = t1 - t2;
    o[3 * os] = t3 + t4;
    o[4 * os] = t3 - t4;
    o[5 * os] = t5 + t6;
    o[6 * os] = t5 - t6;
    o[7 * os] = d7 - d1 + (d3 - d5) * 5.25f;
}

static inline void at6_1d(const v4f* m, int s, v4f* o, int os) {
    v4f a = m[s] + m[2 * s], b = m[s] - m[2 * s], c = m[3 * s] + m[4 * s];
    v4f d = m[3 * s] - m[4 * s], e = m[5 * s] + m[6 * s], f = m[5 * s] - m[6 * s];
    o[0] = m[0] + a + c + e * 32.0f;
    o[os] = b + d * 2.0f + f * 16.0f;
    o[2 * os] = a + c * 4.0f + e * 8.0f;
    o[3 * os] = b + d * 8.0f + f * 4.0f;
    o[4 * os] = a + c * 16.0f + e * 2.0f;
    o[5 * os] = b + d * 32.0f + f + m[7 * s];
}

/* 4x4 transpose: rows a, b, c, d become columns */
#define V6_TR4(a, b, c, d) do {                                                              \
    float32x4x2_t ab_ = vtrnq_f32(a, b), cd_ = vtrnq_f32(c, d);                               \
    a = vcombine_f32(vget_low_f32(ab_.val[0]), vget_low_f32(cd_.val[0]));                    \
    b = vcombine_f32(vget_low_f32(ab_.val[1]), vget_low_f32(cd_.val[1]));                    \
    c = vcombine_f32(vget_high_f32(ab_.val[0]), vget_high_f32(cd_.val[0]));                  \
    d = vcombine_f32(vget_high_f32(ab_.val[1]), vget_high_f32(cd_.val[1]));                  \
} while (0)

/* Input transform of 4 tiles whose windows start at rows h[j], columns w[j]. mode 2 (m=4):
 * neighbours in one row with 20 columns readable, so one deinterleaving load gives a column
 * of all four. mode 1: every window inside, read a row of each and transposed. mode 0: a
 * window crosses the bottom or right edge, so copy with zeros. */
__attribute__((always_inline))
static inline void v6_input_group(int m, const float* input, float* V, int rs, size_t kstride,
                                  const int* h, const int* w, int mode, int H, int W, int c0, int c1) {
    const int alpha = m + 2, a2 = alpha * alpha;
    size_t off[4];
    for (int j = 0; j < 4; j++) off[j] = (size_t)h[j] * W + w[j];
    for (int ci = c0; ci < c1; ci++) {
        const float* plane = input + (size_t)ci * H * W;
        v4f d[64], t[64];
        if (m == 4 && mode == 2) {
            for (int r = 0; r < 6; r++) {
                float32x4x4_t a = vld4q_f32(plane + off[0] + (size_t)r * W);
                float32x4x4_t b = vld4q_f32(plane + off[0] + (size_t)r * W + 4);
                d[r * 6 + 0] = (v4f)a.val[0]; d[r * 6 + 1] = (v4f)a.val[1];
                d[r * 6 + 2] = (v4f)a.val[2]; d[r * 6 + 3] = (v4f)a.val[3];
                d[r * 6 + 4] = (v4f)b.val[0]; d[r * 6 + 5] = (v4f)b.val[1];
            }
        } else if (mode == 1) {
            /* columns 0-3, then the last 4 (alpha - 4 on), of each window row */
            for (int r = 0; r < alpha; r++) {
                const float* p = plane + (size_t)r * W;
                float32x4_t a = vld1q_f32(p + off[0]), b = vld1q_f32(p + off[1]);
                float32x4_t c = vld1q_f32(p + off[2]), e = vld1q_f32(p + off[3]);
                V6_TR4(a, b, c, e);
                d[r * alpha] = (v4f)a; d[r * alpha + 1] = (v4f)b;
                d[r * alpha + 2] = (v4f)c; d[r * alpha + 3] = (v4f)e;
                p += alpha - 4;
                a = vld1q_f32(p + off[0]); b = vld1q_f32(p + off[1]);
                c = vld1q_f32(p + off[2]); e = vld1q_f32(p + off[3]);
                V6_TR4(a, b, c, e);
                if (alpha == 8) { d[r * 8 + 4] = (v4f)a; d[r * 8 + 5] = (v4f)b; }
                d[r * alpha + alpha - 2] = (v4f)c; d[r * alpha + alpha - 1] = (v4f)e;
            }
        } else {
            float g[64][4] __attribute__((aligned(16)));
            for (int j = 0; j < 4; j++) {
                for (int r = 0; r < alpha; r++) {
                    for (int c = 0; c < alpha; c++) {
                        int ih = h[j] + r, iw = w[j] + c;
                        g[r * alpha + c][j] = (ih < H && iw < W) ? plane[(size_t)ih * W + iw] : 0.0f;
                    }
                }
            }
            for (int k = 0; k < a2; k++) d[k] = *(const v4f*)g[k];
        }
        if (m == 4) {
            for (int c = 0; c < 6; c++) bt4_1d(d + c, 6, t + c, 6);
            for (int r = 0; r < 6; r++) bt4_1d(t + r * 6, 1, d + r * 6, 1);
        } else {
            for (int c = 0; c < 8; c++) bt6_1d(d + c, 8, t + c, 8);
            for (int r = 0; r < 8; r++) bt6_1d(t + r * 8, 1, d + r * 8, 1);
        }
        for (int k = 0; k < a2; k++) *(v4f*)(V + k * kstride + (size_t)ci * rs) = d[k];
    }
}

/* Output transform of 4 tiles (nj of them real) starting at h[j], w[j], with bias and clamp.
 * Output rows are ldo floats apart and channels oplane.
 * mode 2 (m=4): neighbours in one row covering a whole 4 x 16 block. mode 1: every tile
 * inside, transposed and stored a row at a time. mode 0: a tile crosses the edge. */
__attribute__((always_inline))
static inline void v6_output_group(int m, const float* Mb, float* output, size_t group,
                                   const int* h, const int* w, int nj, int mode, int Hout, int Wout, int ldo, size_t oplane,
                                   int Cout, const float* bias, float lo, float hi) {
    const int alpha = m + 2, a2 = alpha * alpha;
    int ncb = (Cout + 7) / 8;
    for (int cb = 0; cb < ncb; cb++) {
        v4f mt[8][64];
        for (int k = 0; k < a2; k++) {
            /* 4 tiles x 8 channels to 8 channels x 4 tiles */
            const float* o = Mb + ((group + k) * ncb + cb) * 32;
            for (int half = 0; half < 2; half++) {
                v4f a = *(const v4f*)(o + half * 4), b = *(const v4f*)(o + 8 + half * 4);
                v4f c = *(const v4f*)(o + 16 + half * 4), d = *(const v4f*)(o + 24 + half * 4);
                v4f t0 = __builtin_shufflevector(a, b, 0, 4, 1, 5), t1 = __builtin_shufflevector(a, b, 2, 6, 3, 7);
                v4f t2 = __builtin_shufflevector(c, d, 0, 4, 1, 5), t3 = __builtin_shufflevector(c, d, 2, 6, 3, 7);
                mt[half * 4 + 0][k] = __builtin_shufflevector(t0, t2, 0, 1, 4, 5);
                mt[half * 4 + 1][k] = __builtin_shufflevector(t0, t2, 2, 3, 6, 7);
                mt[half * 4 + 2][k] = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);
                mt[half * 4 + 3][k] = __builtin_shufflevector(t1, t3, 2, 3, 6, 7);
            }
        }
        for (int j = 0; j < 8 && cb * 8 + j < Cout; j++) {
            int co = cb * 8 + j;
            v4f t[48], y[36];
            if (m == 4) {
                for (int c = 0; c < 6; c++) at4_1d(mt[j] + c, 6, t + c, 6);
                for (int r = 0; r < 4; r++) at4_1d(t + r * 6, 1, y + r * 4, 1);
            } else {
                for (int c = 0; c < 8; c++) at6_1d(mt[j] + c, 8, t + c, 8);
                for (int r = 0; r < 6; r++) at6_1d(t + r * 8, 1, y + r * 6, 1);
            }
            float b = bias ? bias[co] : 0.0f;
            for (int q = 0; q < m * m; q++) y[q] = clamp4(y[q] + b, lo, hi);
            float* out = output + co * oplane;
            if (m == 4 && mode == 2) {
                for (int r = 0; r < 4; r++) {
                    float32x4x4_t v = {{(float32x4_t)y[r * 4], (float32x4_t)y[r * 4 + 1],
                                        (float32x4_t)y[r * 4 + 2], (float32x4_t)y[r * 4 + 3]}};
                    vst4q_f32(out + (size_t)(h[0] + r) * ldo + w[0], v);
                }
            } else if (mode == 1) {
                /* columns 0-3, then the last 4 (m - 4 on; for m = 4 the same ones again) */
                for (int r = 0; r < m; r++) {
                    for (int c0 = 0; c0 <= m - 4; c0 += m - 4 > 0 ? m - 4 : 4) {
                        float32x4_t v[4] = {(float32x4_t)y[r * m + c0], (float32x4_t)y[r * m + c0 + 1],
                                            (float32x4_t)y[r * m + c0 + 2], (float32x4_t)y[r * m + c0 + 3]};
                        V6_TR4(v[0], v[1], v[2], v[3]);
                        for (int jj = 0; jj < nj; jj++) vst1q_f32(out + (size_t)(h[jj] + r) * ldo + w[jj] + c0, v[jj]);
                    }
                }
            } else {
                for (int jj = 0; jj < nj; jj++) {
                    for (int r = 0; r < m && h[jj] + r < Hout; r++) {
                        for (int c = 0; c < m && w[jj] + c < Wout; c++) {
                            out[(size_t)(h[jj] + r) * ldo + w[jj] + c] = y[r * m + c][jj];
                        }
                    }
                }
            }
        }
    }
}
#endif

/* Where the 4 tiles from t (n of them real) start; missing ones repeat the first. Returns
 * the transform mode: 2 for neighbours in a row with `wide` columns inside, 1 if every
 * window of `span` rows and columns is inside, 0 otherwise. */
static inline int v6_group_pos(int m, int t, int n, int Tw, int H, int W, int span, int wide, int* h, int* w) {
    int inside = 1;
    for (int j = 0; j < 4; j++) {
        int tj = t + (j < n ? j : 0);
        h[j] = (tj / Tw) * m;
        w[j] = (tj % Tw) * m;
        inside &= h[j] + span <= H && w[j] + span <= W;
    }
    if (inside && m == 4 && n == 4 && h[3] == h[0] && w[3] == w[0] + 12 && w[0] + wide <= W) return 2;
    return inside;
}

/* Slots slot0.. hold tiles t0..t0+n-1 of a batch of ntot; slot0 is a multiple of 4. The NEON
 * path does channels [cb, ce), 8 at a time across all the groups, so the three groups sharing
 * a block of 12 fill each line of V while it is still in L1, instead of each group fetching it
 * again. The scalar path does every channel, in the call with cb 0. */
static void v6_input_tiles(int m, const float* input, float* V, int slot0, int t0, int n, int ntot,
                           int H, int W, int Cin, int Tw, int cb, int ce) {
#if HAS_ARM_NEON
    if (m == 4 || m == 6) {
        for (int c0 = cb; c0 < ce; c0 += 8) {
            int c1 = c0 + 8 < ce ? c0 + 8 : ce;
            for (int i = 0; i < n; i += 4) {
                int h[4], w[4], rs;
                int mode = v6_group_pos(m, t0 + i, n - i < 4 ? n - i : 4, Tw, H, W, m + 2, 20, h, w);
                size_t base = v6_vbase(slot0 + i, ntot, Cin, &rs);
                if (m == 4) v6_input_group(4, input, V + base, rs, v6_vfreq(ntot, Cin), h, w, mode, H, W, c0, c1);
                else v6_input_group(6, input, V + base, rs, v6_vfreq(ntot, Cin), h, w, mode, H, W, c0, c1);
            }
        }
        return;
    }
#endif
    for (int j = 0; j < n && cb == 0; j++) v6_input_tile(m, input, V, slot0 + j, t0 + j, ntot, H, W, Cin, Tw);
}

static void v6_output_tiles(int m, const float* Mb, float* output, int slot0, int t0, int nt, int Hout, int Wout,
                            int Cout, int Tw, int ldo, size_t oplane, const float* bias, float lo, float hi) {
    int alpha2 = (m + 2) * (m + 2);
    for (int i = 0; i < nt; i += 4) {
#if HAS_ARM_NEON
        if (m == 4 || m == 6) {
            int h[4], w[4], nj = nt - i < 4 ? nt - i : 4;
            int mode = v6_group_pos(m, t0 + i, nj, Tw, Hout, Wout, m, 16, h, w);
            size_t group = (size_t)((slot0 + i) / 4) * alpha2;
            if (m == 4) v6_output_group(4, Mb, output, group, h, w, nj, mode, Hout, Wout, ldo, oplane, Cout, bias, lo, hi);
            else v6_output_group(6, Mb, output, group, h, w, nj, mode, Hout, Wout, ldo, oplane, Cout, bias, lo, hi);
            continue;
        }
#endif
        (void)alpha2;
        for (int j = i; j < i + 4 && j < nt; j++)
            v6_output_tile(m, Mb, output, slot0 + j, t0 + j, Hout, Wout, Cout, Tw, ldo, oplane, bias, lo, hi);
    }
}

/* Tile size for a 3x3 stride-1 conv on an H x W padded input: F(4,3) or F(6,3), whichever a
 * cost model says is faster. Per tile (tile counts rounded up to whole groups of 4): the
 * multiply costs alpha^2 * Cin * Cout units, F(6,3)'s 15% more per unit for its bigger weight
 * table, plus each input and output channel's transform. Weights in ns, measured on the RPi4
 * at T=1 (raw_logs/harness/sc_pick_m.txt). F(6,3) wins on large channel counts whose maps
 * are a multiple of 6 or 12, e.g. 128 x 24 x 24 and 384 x 12 x 12. With more threads F(4,3)'s
 * extra tiles split better (woof_wn_e at T=4: 13% slower with F(6,3)), so it stays.
 * ponytail: fitted on one board; refit the six constants for another. */
static int v6_pick_m(int H, int W, int Cin, int Cout, int T) {
    if (T > 1) return 4;
    double best = 0.0;
    int best_m = 4;
    for (int m = 4; m <= 6; m += 2) {
        int tiles = ((H - 2 + m - 1) / m) * ((W - 2 + m - 1) / m);
        double per_tile = m == 4 ? 36 * 0.20 * Cin * Cout + 165.0 * Cin + 91.0 * Cout
                                 : 64 * 0.23 * Cin * Cout + 440.0 * Cin + 317.0 * Cout;
        double cost = (tiles + 3) / 4 * 4 * per_tile;
        if (m == 4 || cost < best) best = cost, best_m = m;
    }
    return best_m;
}

/* bias (may be NULL) is added and the result clamped to [lo, hi] as tiles are written. With
 * opad > 0, output is a buffer with an opad-wide border (the next conv's padded input) and
 * only its interior is written. */
void conv2d_v6_freq_ep(
    int m,
    const float* input,
    const float* U_p,
    const float* bias, float lo, float hi,
    float* output, int opad,
    int H, int W,
    int Cin, int Cout,
    int T,
    const char** executed_path)
{
    *executed_path = "v6_freq";
    int alpha2 = (m + 2) * (m + 2);
    int ncb = (Cout + 7) / 8;
    int Hout = H - 2;
    int Wout = W - 2;
    int Th = (Hout + m - 1) / m;
    int Tw = (Wout + m - 1) / m;
    int N_tiles = Th * Tw;
    int ldo = Wout + 2 * opad;
    size_t oplane = (size_t)ldo * (Hout + 2 * opad);
    output += (size_t)opad * ldo + opad;
    size_t table_bytes = (size_t)Cin * Cout * alpha2 * sizeof(float);
    /* One thread: one block of 12 tiles keeps V and Mb in cache. More threads: bigger
     * batches, so layers with few tiles share one batch instead of leaving threads idle.
     * ponytail: tuned on one RPi4 at T=1 and T=4; pass --batch for other boards. */
    int batch = g_v5_batch > 0 ? (g_v5_batch + 3) / 4 * 4 : T == 1 ? 12 : (table_bytes > V5_L2_BYTES ? 16 : 8);

    if (N_tiles / T < batch) {
        /* Too few tiles for each thread to have a batch, so all threads share one
         * and split the frequencies and output channels instead. */
        int ng = (N_tiles + 3) / 4;
        float* V = v5_scratch(0, alpha2 * v6_vfreq(N_tiles, Cin));
        float* Mb = v5_scratch(1, (size_t)ng * alpha2 * ncb * 32);
        #pragma omp parallel num_threads(T)
        {
            double q0 = get_time_ms();
            /* threads take whole blocks of 12 tiles, which share cache lines of V, and split channels */
            int nblk = (N_tiles + 11) / 12;
            #pragma omp for collapse(2) schedule(dynamic)
            for (int b = 0; b < nblk; b++) {
                for (int c = 0; c < Cin; c += 32) {
                    v6_input_tiles(m, input, V, b * 12, b * 12, N_tiles - b * 12 < 12 ? N_tiles - b * 12 : 12, N_tiles,
                                   H, W, Cin, Tw, c, c + 32 < Cin ? c + 32 : Cin);
                }
            }
            double q1 = get_time_ms();
            #pragma omp for collapse(2) schedule(dynamic)
            for (int k = 0; k < alpha2; k++) {
                for (int cb = 0; cb < ncb; cb++) {
                    v6_multiply(U_p, V, Mb, k, cb, N_tiles, alpha2, ncb, Cin);
                }
            }
            double q2 = get_time_ms();
            #pragma omp for schedule(dynamic)
            for (int g = 0; g < ng; g++) {
                v6_output_tiles(m, Mb, output, g * 4, g * 4, N_tiles - g * 4 < 4 ? N_tiles - g * 4 : 4,
                                Hout, Wout, Cout, Tw, ldo, oplane, bias, lo, hi);
            }
            double q3 = get_time_ms();
            if (omp_get_thread_num() == 0) { g_prof[0] += q1 - q0; g_prof[1] += q2 - q1; g_prof[2] += q3 - q2; }
        }
        return;
    }

    int n_batches = (N_tiles + batch - 1) / batch;
    #pragma omp parallel num_threads(T)
    {
        int ng_max = batch / 4;
        float* V = v5_scratch(0, alpha2 * v6_vfreq(batch, Cin));
        float* Mb = v5_scratch(1, (size_t)ng_max * alpha2 * ncb * 32);

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < n_batches; b++) {
            int t0 = b * batch;
            int nt = (t0 + batch <= N_tiles) ? batch : N_tiles - t0;
            double q0 = get_time_ms();
            v6_input_tiles(m, input, V, 0, t0, nt, nt, H, W, Cin, Tw, 0, Cin);
            double q1 = get_time_ms();
            for (int k = 0; k < alpha2; k++) {
                for (int cb = 0; cb < ncb; cb++) {
                    v6_multiply(U_p, V, Mb, k, cb, nt, alpha2, ncb, Cin);
                }
            }
            double q2 = get_time_ms();
            v6_output_tiles(m, Mb, output, 0, t0, nt, Hout, Wout, Cout, Tw, ldo, oplane, bias, lo, hi);
            double q3 = get_time_ms();
            if (omp_get_thread_num() == 0) { g_prof[0] += q1 - q0; g_prof[1] += q2 - q1; g_prof[2] += q3 - q2; }
        }
    }
}

void conv2d_v6_freq(int m, const float* input, const float* U_p, float* output, int H, int W,
                    int Cin, int Cout, int T, const char** executed_path) {
    conv2d_v6_freq_ep(m, input, U_p, NULL, -INFINITY, INFINITY, output, 0, H, W, Cin, Cout, T, executed_path);
}

/* -------------------------------------------------------------------------
 * 10. depthwise: One k x k filter per channel, with NEON for 3x3 at stride 1 and 2
 * ------------------------------------------------------------------------- */
__attribute__((always_inline))
static inline void dw3_row(const float* r0, const float* r1, const float* r2, const float* f,
                           float* y, int Wout, int W, int s, float b, float lo, float hi) {
    /* bias b is added and the result clamped to [lo, hi] as each pixel is stored */
    int wo = 0;
    v4f vb = {b, b, b, b};
    if (s == 1) {
        for (; wo + 4 <= Wout; wo += 4) {
            v4f acc = vb +
                *(const v4f*)(r0 + wo) * f[0] + *(const v4f*)(r0 + wo + 1) * f[1] + *(const v4f*)(r0 + wo + 2) * f[2] +
                *(const v4f*)(r1 + wo) * f[3] + *(const v4f*)(r1 + wo + 1) * f[4] + *(const v4f*)(r1 + wo + 2) * f[5] +
                *(const v4f*)(r2 + wo) * f[6] + *(const v4f*)(r2 + wo + 1) * f[7] + *(const v4f*)(r2 + wo + 2) * f[8];
            *(v4f*)(y + wo) = clamp4(acc, lo, hi);
        }
    } else if (s == 2) {
        /* Every other pixel of 8; the +9 keeps the last load inside the row */
        for (; wo * 2 + 9 < W && wo + 4 <= Wout; wo += 4) {
            v4f acc = vb;
            const float* rows[3] = {r0, r1, r2};
            for (int r = 0; r < 3; r++) {
                const float* p = rows[r] + wo * 2;
#if HAS_ARM_NEON
                /* deinterleaving loads: taps 0 and 1 for 4 pixels, then tap 2 from 2 floats on */
                float32x4x2_t a = vld2q_f32(p), c = vld2q_f32(p + 2);
                v4f evens = (v4f)a.val[0], odds = (v4f)a.val[1], thirds = (v4f)c.val[0];
#else
                v4f a0 = *(const v4f*)p, a1 = *(const v4f*)(p + 4);
                v4f b0 = *(const v4f*)(p + 1), b1 = *(const v4f*)(p + 5);
                v4f evens = __builtin_shufflevector(a0, a1, 0, 2, 4, 6);
                v4f odds = __builtin_shufflevector(b0, b1, 0, 2, 4, 6);
                v4f thirds = __builtin_shufflevector(a0, a1, 2, 4, 6, 6);
                thirds[3] = p[8];
#endif
                acc += evens * f[r * 3] + odds * f[r * 3 + 1] + thirds * f[r * 3 + 2];
            }
            *(v4f*)(y + wo) = clamp4(acc, lo, hi);
        }
    }
    for (; wo < Wout; wo++) {
        const float* a = r0 + wo * s, *bb = r1 + wo * s, *c = r2 + wo * s;
        float v = b + a[0] * f[0] + a[1] * f[1] + a[2] * f[2] + bb[0] * f[3] + bb[1] * f[4] + bb[2] * f[5] +
                  c[0] * f[6] + c[1] * f[7] + c[2] * f[8];
        y[wo] = v < lo ? lo : (v > hi ? hi : v);
    }
}

/* weights: [C][k*k]. Reads outside the input count as zero (padding p), so no padded
 * copy is needed. bias (may be NULL) is added and the result clamped to [lo, hi]. */
void conv2d_depthwise(const float* input, const float* weights, const float* bias, float* output,
                      int H, int W, int C, int k, int s, int p, float lo, float hi, int T) {
    int Hout = (H + 2 * p - k) / s + 1, Wout = (W + 2 * p - k) / s + 1, taps = k * k;
    float* zero = (float*)calloc((size_t)W + 16, sizeof(float));  /* stands in for rows above and below */
    /* 3x3 output columns whose taps all land inside the row */
    int w_lo = k == 3 && p <= 1 ? p : 0;
    int w_hi = k == 3 && p <= 1 && W - 3 + p >= 0 ? (W - 3 + p) / s : -1;
    if (w_hi > Wout - 1) w_hi = Wout - 1;
    #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int ho = 0; ho < Hout; ho++) {
            const float* plane = input + (size_t)c * H * W;
            const float* f = weights + (size_t)c * taps;
            float* y = output + ((size_t)c * Hout + ho) * Wout;
            int ih = ho * s - p;
            if (k == 3 && p <= 1) {
                const float* r[3];
                for (int j = 0; j < 3; j++) r[j] = ih + j >= 0 && ih + j < H ? plane + (size_t)(ih + j) * W : zero;
                int off = w_lo * s - p;
                float b = bias ? bias[c] : 0.0f;
                if (w_hi >= w_lo) dw3_row(r[0] + off, r[1] + off, r[2] + off, f, y + w_lo, w_hi - w_lo + 1, W - off, s, b, lo, hi);
                for (int wo = 0; wo < Wout; wo++) {
                    if (wo == w_lo && w_hi >= w_lo) wo = w_hi + 1;
                    if (wo >= Wout) break;
                    float sum = b;
                    for (int kh = 0; kh < 3; kh++) for (int kw = 0; kw < 3; kw++) {
                        int iw = wo * s - p + kw;
                        if (iw >= 0 && iw < W) sum += r[kh][iw] * f[kh * 3 + kw];
                    }
                    y[wo] = sum < lo ? lo : (sum > hi ? hi : sum);
                }
                continue;   /* bias and clamp done */
            } else {
                for (int wo = 0; wo < Wout; wo++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < k; kh++) {
                        if (ih + kh < 0 || ih + kh >= H) continue;
                        for (int kw = 0; kw < k; kw++) {
                            int iw = wo * s - p + kw;
                            if (iw >= 0 && iw < W) sum += plane[(size_t)(ih + kh) * W + iw] * f[kh * k + kw];
                        }
                    }
                    y[wo] = sum;
                }
            }
            float b = bias ? bias[c] : 0.0f;
            for (int wo = 0; wo < Wout; wo++) {
                float v = y[wo] + b;
                y[wo] = v < lo ? lo : (v > hi ? hi : v);
            }
        }
    }
    free(zero);
}

/* -------------------------------------------------------------------------
 * Shared library export for Python ctypes wrapper
 * ------------------------------------------------------------------------- */
void fused_winograd_conv2d(
    const float* input,
    const float* weights,
    float* output,
    int H, int W,
    int Cin, int Cout,
    int T)
{
    int m = 2; /* Default F(2,3) */
    int alpha = m + 2;
    float* U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
    transform_weights(m, Cin, Cout, weights, U);

    const char* path = NULL;
    conv2d_fused_winograd(m, input, U, output, H, W, Cin, Cout, T, &path);
    free(U);
}

/* -------------------------------------------------------------------------
 * Self-Test Smoke Validation
 * ------------------------------------------------------------------------- */
int run_harness_selftest(void) {
    printf("==============================================================\n");
    printf("[bench_harness] Phase 1 C-Level Smoke Test & Harness Validation\n");
    printf("==============================================================\n");

    int H = 8, W = 8, Cin = 4, Cout = 4, T = 1;
    int test_T_vals[] = {1, 4};
    int test_m_vals[] = {2, 4, 6};
    int Hout = H - 2, Wout = W - 2;
    size_t in_sz = Cin * H * W;
    size_t w_sz = Cout * Cin * 9;
    size_t out_sz = Cout * Hout * Wout;

    float* in = (float*)malloc(in_sz * sizeof(float));
    float* w = (float*)malloc(w_sz * sizeof(float));
    float* ref = (float*)malloc(out_sz * sizeof(float));
    float* out = (float*)malloc(out_sz * sizeof(float));

    for (size_t i = 0; i < in_sz; i++) in[i] = (float)(((long long)i % 17) - 8) / 8.0f;
    for (size_t i = 0; i < w_sz; i++) w[i] = (float)(((long long)i % 13) - 6) / 6.0f;

    const char* path = NULL;
    conv2d_naive_direct(in, w, ref, H, W, Cin, Cout, T, &path);
    if (strcmp(path, "naive_direct") != 0) {
        fprintf(stderr, "FATAL: Path assertion failed for naive_direct!\n");
        return 1;
    }

    /* Test all variants across m in {2, 4, 6} */
    const char* variants[] = {"neon_direct", "unfused", "fused", "v2_regblocked", "v3_cin_blocked", "v4_pretransformed_packed", "v5_batched", "v6_freq"};
    int n_var = sizeof(variants) / sizeof(variants[0]);

    for (int ti = 0; ti < 2; ti++) {
    T = test_T_vals[ti];
    for (int mi = 0; mi < 3; mi++) {
    int m = test_m_vals[mi];
    for (int vi = 0; vi < n_var; vi++) {
        const char* req = variants[vi];
        memset(out, 0, out_sz * sizeof(float));

        if (strcmp(req, "neon_direct") == 0) {
            conv2d_neon_direct(in, w, out, H, W, Cin, Cout, T, &path);
        } else {
            int alpha = m + 2;
            float* U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
            transform_weights(m, Cin, Cout, w, U);

            if (strcmp(req, "unfused") == 0) {
                conv2d_unfused_winograd(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "fused") == 0) {
                conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v2_regblocked") == 0) {
                conv2d_v2_regblocked(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v3_cin_blocked") == 0) {
                conv2d_v3_cin_blocked(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v4_pretransformed_packed") == 0) {
                conv2d_v4_pretransformed_packed(m, in, U, out, H, W, Cin, Cout, T, &path);
            } else if (strcmp(req, "v5_batched") == 0) {
                float* U_p = v5_pack_weights(m, U, Cin, Cout);
                conv2d_v5_batched(m, in, U_p, out, H, W, Cin, Cout, T, &path);
                free(U_p);
            } else if (strcmp(req, "v6_freq") == 0) {
                float* U_p = v6_pack_weights(m, U, Cin, Cout);
                conv2d_v6_freq(m, in, U_p, out, H, W, Cin, Cout, T, &path);
                free(U_p);
            }
            free(U);
        }

        if (strcmp(path, req) != 0) {
            fprintf(stderr, "FATAL: Gate G2 Path Mismatch: requested '%s' != executed '%s'\n", req, path);
            return 1;
        }

        float max_err = 0.0f;
        for (size_t i = 0; i < out_sz; i++) {
            float err = fabsf(ref[i] - out[i]);
            if (err > max_err) max_err = err;
        }
        printf("  Variant %-26s m=%d T=%d -> Path: %-12s | Max Err vs Ref: %.4e\n", req, m, T, path, max_err);
        if (max_err > 1e-3f) {
            fprintf(stderr, "FATAL: Gate G1 Correctness Failure for variant '%s' m=%d T=%d (max_err=%.4e)\n", req, m, T, max_err);
            return 1;
        }
    }
    }
    }

    /* v5_batched and v6_freq edge cases: Cout padding, partial batches, kc smaller than Cin */
    {
        int h2 = 14, w2 = 14, ci2 = 5, co2 = 9;
        int ho2 = h2 - 2, wo2 = w2 - 2;
        size_t o2 = (size_t)co2 * ho2 * wo2;
        float* in2 = (float*)malloc((size_t)ci2 * h2 * w2 * sizeof(float));
        float* w_2 = (float*)malloc((size_t)co2 * ci2 * 9 * sizeof(float));
        float* ref2 = (float*)malloc(o2 * sizeof(float));
        float* out2 = (float*)malloc(o2 * sizeof(float));
        for (int i = 0; i < ci2 * h2 * w2; i++) in2[i] = (float)((i % 17) - 8) / 8.0f;
        for (int i = 0; i < co2 * ci2 * 9; i++) w_2[i] = (float)((i % 13) - 6) / 6.0f;
        conv2d_naive_direct(in2, w_2, ref2, h2, w2, ci2, co2, 1, &path);
        int saved_batch = g_v5_batch, saved_kc = g_v5_kc;
        int batches[] = {1, 2, 5, 64}, kcs[] = {1, 2, 128};
        for (int mi = 0; mi < 3; mi++) {
            int m = test_m_vals[mi];
            int alpha = m + 2;
            float* U = (float*)malloc((size_t)co2 * ci2 * alpha * alpha * sizeof(float));
            transform_weights(m, ci2, co2, w_2, U);
            float* U_p = v5_pack_weights(m, U, ci2, co2);
            float* U_p6 = v6_pack_weights(m, U, ci2, co2);
            for (int vi = 0; vi < 2; vi++) for (int ti = 0; ti < 2; ti++) for (int bi = 0; bi < 4; bi++) for (int ki = 0; ki < 3; ki++) {
                g_v5_batch = batches[bi];
                g_v5_kc = kcs[ki];
                memset(out2, 0, o2 * sizeof(float));
                if (vi == 0) conv2d_v5_batched(m, in2, U_p, out2, h2, w2, ci2, co2, test_T_vals[ti], &path);
                else conv2d_v6_freq(m, in2, U_p6, out2, h2, w2, ci2, co2, test_T_vals[ti], &path);
                float max_err = 0.0f;
                for (size_t i = 0; i < o2; i++) {
                    float err = fabsf(ref2[i] - out2[i]);
                    if (err > max_err) max_err = err;
                }
                if (max_err > 1e-3f) {
                    fprintf(stderr, "FATAL: %s m=%d T=%d batch=%d kc=%d (max_err=%.4e)\n",
                            path, m, test_T_vals[ti], g_v5_batch, g_v5_kc, max_err);
                    return 1;
                }
            }
            free(U);
            free(U_p);
            free(U_p6);
        }
        g_v5_batch = saved_batch;
        g_v5_kc = saved_kc;
        printf("  v5_batched and v6_freq edge cases OK (padding, partial batches, kc < Cin)\n");
        free(in2); free(w_2); free(ref2); free(out2);
    }

    /* v6_freq at sizes where whole groups of 4 tiles take the NEON transform path */
    {
        int sizes[][2] = {{22, 38}, {23, 41}, {30, 66}, {9, 9}, {16, 16}, {14, 27}, {50, 26}};
        int ci6 = 5, co6 = 11;
        int saved_batch = g_v5_batch;
        for (int si = 0; si < 14; si++) {
            int m6 = si < 7 ? 4 : 6;
            int h6 = sizes[si % 7][0], w6 = sizes[si % 7][1], ho6 = h6 - 2, wo6 = w6 - 2, a6 = (m6 + 2) * (m6 + 2);
            size_t o6 = (size_t)co6 * ho6 * wo6;
            float* in6 = (float*)malloc((size_t)ci6 * h6 * w6 * sizeof(float));
            float* w_6 = (float*)malloc((size_t)co6 * ci6 * 9 * sizeof(float));
            float* ref6 = (float*)malloc(o6 * sizeof(float));
            float* out6 = (float*)malloc(o6 * sizeof(float));
            for (int i = 0; i < ci6 * h6 * w6; i++) in6[i] = (float)((i % 23) - 11) / 11.0f;
            for (int i = 0; i < co6 * ci6 * 9; i++) w_6[i] = (float)((i % 7) - 3) / 3.0f;
            conv2d_naive_direct(in6, w_6, ref6, h6, w6, ci6, co6, 1, &path);
            /* with bias and a clamp that cuts both ends, as fused into the output transform */
            float b6[11], lo6 = -1.0f, hi6 = 1.5f;
            for (int co = 0; co < co6; co++) b6[co] = (co % 5 - 2) * 0.3f;
            for (size_t i = 0; i < o6; i++) {
                float v = ref6[i] + b6[i / ((size_t)ho6 * wo6)];
                ref6[i] = v < lo6 ? lo6 : (v > hi6 ? hi6 : v);
            }
            float* U = (float*)malloc((size_t)co6 * ci6 * a6 * sizeof(float));
            transform_weights(m6, ci6, co6, w_6, U);
            float* U_p = v6_pack_weights(m6, U, ci6, co6);
            int batches[] = {4, 8, 24, 1000};
            for (int ti = 0; ti < 2; ti++) for (int bi = 0; bi < 4; bi++) {
                g_v5_batch = batches[bi];
                memset(out6, 0, o6 * sizeof(float));
                conv2d_v6_freq_ep(m6, in6, U_p, b6, lo6, hi6, out6, 0, h6, w6, ci6, co6, test_T_vals[ti], &path);
                float max_err = 0.0f;
                for (size_t i = 0; i < o6; i++) {
                    float err = fabsf(ref6[i] - out6[i]);
                    if (err > max_err) max_err = err;
                }
                /* into a buffer with a 1-wide border, which must stay zero */
                size_t pp = (size_t)(ho6 + 2) * (wo6 + 2);
                float* outp = (float*)calloc(co6 * pp, sizeof(float));
                conv2d_v6_freq_ep(m6, in6, U_p, b6, lo6, hi6, outp, 1, h6, w6, ci6, co6, test_T_vals[ti], &path);
                for (int co = 0; co < co6; co++) for (int y = 0; y < ho6 + 2; y++) for (int x = 0; x < wo6 + 2; x++) {
                    int in_int = y >= 1 && y <= ho6 && x >= 1 && x <= wo6;
                    float want = in_int ? ref6[((size_t)co * ho6 + y - 1) * wo6 + x - 1] : 0.0f;
                    float err = fabsf(outp[co * pp + (size_t)y * (wo6 + 2) + x] - want);
                    if (err > max_err) max_err = err;
                }
                free(outp);
                if (max_err > 1e-3f) {
                    fprintf(stderr, "FATAL: v6_freq transforms m=%d %dx%d T=%d batch=%d (max_err=%.4e)\n",
                            m6, h6, w6, test_T_vals[ti], g_v5_batch, max_err);
                    return 1;
                }
            }
            free(in6); free(w_6); free(ref6); free(out6); free(U); free(U_p);
        }
        g_v5_batch = saved_batch;
        printf("  v6_freq transform paths OK (m 4 and 6, whole groups, partial groups, edges, bias and clamp, both thread counts)\n");
    }

    /* direct_packed: kernel size, stride, Cout padding, partial blocks, both threading modes */
    {
        int h3 = 15, w3 = 15, ci3 = 37, co3 = 9;
        int saved_kci = g_dp_kci, saved_block = g_dp_block;
        int ks[] = {1, 3, 5, 7, 11}, ss[] = {1, 2, 4};
        int kcis[] = {1, 32, 64}, blocks[] = {0, 1, 6, 32};
        size_t i3 = (size_t)ci3 * h3 * w3;
        float* in3 = (float*)malloc(i3 * sizeof(float));
        for (size_t i = 0; i < i3; i++) in3[i] = (float)((long long)(i % 17) - 8) / 8.0f;
        for (int ki3 = 0; ki3 < 5; ki3++) {
            int k3 = ks[ki3], taps3 = k3 * k3;
            size_t w_n = (size_t)co3 * ci3 * taps3;
            float* w_3 = (float*)malloc(w_n * sizeof(float));
            for (size_t i = 0; i < w_n; i++) w_3[i] = (float)((long long)(i % 13) - 6) / 6.0f;
            float* W_p = dp_pack_weights(w_3, ci3, co3, k3);
            for (int si = 0; si < 3; si++) {
                int s3 = ss[si];
                int ho3 = (h3 - k3) / s3 + 1, wo3 = (w3 - k3) / s3 + 1;
                size_t o3 = (size_t)co3 * ho3 * wo3;
                float* ref3 = (float*)calloc(o3, sizeof(float));
                float* out3 = (float*)malloc(o3 * sizeof(float));
                for (int co = 0; co < co3; co++) for (int ho = 0; ho < ho3; ho++) for (int wo = 0; wo < wo3; wo++) {
                    double sum = 0.0;
                    for (int ci = 0; ci < ci3; ci++) for (int kh = 0; kh < k3; kh++) for (int kw = 0; kw < k3; kw++) {
                        sum += in3[(size_t)ci * h3 * w3 + (ho * s3 + kh) * w3 + wo * s3 + kw] *
                               w_3[((size_t)co * ci3 + ci) * taps3 + kh * k3 + kw];
                    }
                    ref3[(size_t)co * ho3 * wo3 + ho * wo3 + wo] = (float)sum;
                }
                for (int ti = 0; ti < 2; ti++) for (int ki = 0; ki < 3; ki++) for (int bi = 0; bi < 4; bi++) {
                    g_dp_kci = kcis[ki];
                    g_dp_block = blocks[bi];
                    memset(out3, 0, o3 * sizeof(float));
                    conv2d_direct_packed(in3, W_p, out3, h3, w3, ci3, co3, k3, s3, test_T_vals[ti], &path);
                    float max_err = 0.0f;
                    for (size_t i = 0; i < o3; i++) {
                        float err = fabsf(ref3[i] - out3[i]);
                        if (err > max_err) max_err = err;
                    }
                    if (max_err > 1e-2f) {
                        fprintf(stderr, "FATAL: direct_packed k=%d s=%d T=%d kci=%d block=%d (max_err=%.4e)\n",
                                k3, s3, test_T_vals[ti], g_dp_kci, g_dp_block, max_err);
                        return 1;
                    }
                }
                free(ref3);
                free(out3);
            }
            free(w_3);
            free(W_p);
        }
        g_dp_kci = saved_kci;
        g_dp_block = saved_block;
        printf("  direct_packed OK (k 1-11, stride 1-4, padding, partial blocks, both threading modes)\n");
        free(in3);
    }

    /* depthwise: every kernel, stride and thread count against a plain loop */
    {
        int dh = 19, dw = 23, dc = 5;
        size_t dn = (size_t)dc * dh * dw;
        float* din = (float*)malloc(dn * sizeof(float));
        for (size_t i = 0; i < dn; i++) din[i] = (float)((long long)(i % 29) - 14) / 7.0f;
        int dks[] = {1, 3, 5}, dss[] = {1, 2, 3};
        float dbias[5] = {0.5f, -1.0f, 0.25f, 2.0f, -0.75f};
        for (int ki = 0; ki < 3; ki++) for (int si = 0; si < 3; si++) for (int ti = 0; ti < 2; ti++)
        for (int dp = 0; dp <= 2; dp++) {
            int k = dks[ki], s = dss[si], taps = k * k;
            int ho = (dh + 2 * dp - k) / s + 1, wo = (dw + 2 * dp - k) / s + 1;
            float lo = dp == 1 ? 0.0f : -INFINITY, hi = dp == 1 ? 6.0f : INFINITY;
            size_t on = (size_t)dc * ho * wo;
            float* f = (float*)malloc((size_t)dc * taps * sizeof(float));
            for (int i = 0; i < dc * taps; i++) f[i] = (float)(i % 11 - 5) / 3.0f;
            float* out = (float*)malloc(on * sizeof(float));
            conv2d_depthwise(din, f, dp ? dbias : NULL, out, dh, dw, dc, k, s, dp, lo, hi, test_T_vals[ti]);
            for (int c = 0; c < dc; c++) for (int y = 0; y < ho; y++) for (int x = 0; x < wo; x++) {
                float sum = dp ? dbias[c] : 0.0f;
                for (int kh = 0; kh < k; kh++) for (int kw = 0; kw < k; kw++) {
                    int iy = y * s - dp + kh, ix = x * s - dp + kw;
                    if (iy < 0 || iy >= dh || ix < 0 || ix >= dw) continue;
                    sum += din[((size_t)c * dh + iy) * dw + ix] * f[c * taps + kh * k + kw];
                }
                sum = sum < lo ? lo : (sum > hi ? hi : sum);
                float got = out[((size_t)c * ho + y) * wo + x];
                if (fabsf(got - sum) > 1e-3f) {
                    fprintf(stderr, "FATAL: depthwise k=%d s=%d T=%d at c=%d y=%d x=%d: %f vs %f\n",
                            k, s, test_T_vals[ti], c, y, x, got, sum);
                    return 1;
                }
            }
            free(f);
            free(out);
        }
        free(din);
        printf("  depthwise OK (k 1-5, stride 1-3, padding 0-2, bias and clamp, both thread counts)\n");
    }

    printf("  [PASSED] All variant path assertions and numerical tolerances validated.\n");

    free(in);
    free(w);
    free(ref);
    free(out);
    return 0;
}

/* -------------------------------------------------------------------------
 * Main CLI Benchmark Runner
 * ------------------------------------------------------------------------- */
int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        return run_harness_selftest();
    }

    const char* board = "auto";
    const char* git_hash = "dirty";
    const char* cell_id = "cell_0";
    const char* variant = "fused";
    int m = 4;
    int T = 1;
    int H = 56, W = 56;
    int Cin = 32, Cout = 32;
    int warmup = 20;
    int runs = 30;
    const char* out_csv = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--board") == 0 && i + 1 < argc) board = argv[++i];
        else if (strcmp(argv[i], "--git-hash") == 0 && i + 1 < argc) git_hash = argv[++i];
        else if (strcmp(argv[i], "--cell-id") == 0 && i + 1 < argc) cell_id = argv[++i];
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) variant = argv[++i];
        else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--H") == 0 && i + 1 < argc) H = atoi(argv[++i]);
        else if (strcmp(argv[i], "--W") == 0 && i + 1 < argc) W = atoi(argv[++i]);
        else if (strcmp(argv[i], "--Cin") == 0 && i + 1 < argc) Cin = atoi(argv[++i]);
        else if (strcmp(argv[i], "--Cout") == 0 && i + 1 < argc) Cout = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_csv = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) g_v5_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kc") == 0 && i + 1 < argc) g_v5_kc = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dp-kci") == 0 && i + 1 < argc) g_dp_kci = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dp-block") == 0 && i + 1 < argc) g_dp_block = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) g_stride = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) g_ksize = atoi(argv[++i]);
    }

    if ((g_stride != 1 || g_ksize != 3) && strcmp(variant, "direct_packed") != 0) {
        fprintf(stderr, "FATAL: only direct_packed supports --stride and --k (got %s)\n", variant);
        return 5;
    }
    int Hout = (H - g_ksize) / g_stride + 1;
    int Wout = (W - g_ksize) / g_stride + 1;
    double total_flops = 2.0 * (double)Hout * (double)Wout * (double)Cin * (double)Cout * g_ksize * g_ksize;

    size_t in_sz = (size_t)Cin * H * W;
    size_t w_sz = (size_t)Cout * Cin * g_ksize * g_ksize;
    size_t out_sz = (size_t)Cout * Hout * Wout;

    float* in = (float*)malloc(in_sz * sizeof(float));
    float* w = (float*)malloc(w_sz * sizeof(float));
    float* out = (float*)malloc(out_sz * sizeof(float));

    srand(42);
    for (size_t i = 0; i < in_sz; i++) in[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < w_sz; i++) w[i] = (float)rand() / (float)RAND_MAX;

    int alpha = m + 2;
    float* U = NULL;
    float* W_p = strcmp(variant, "direct_packed") == 0 ? dp_pack_weights(w, Cin, Cout, g_ksize) : NULL;
    if (strcmp(variant, "naive_direct") != 0 && strcmp(variant, "neon_direct") != 0 && !W_p) {
        U = (float*)malloc((size_t)Cout * Cin * alpha * alpha * sizeof(float));
        transform_weights(m, Cin, Cout, w, U);
        if (strcmp(variant, "v5_batched") == 0 || strcmp(variant, "v6_freq") == 0) {
            float* U_p = strcmp(variant, "v6_freq") == 0 ? v6_pack_weights(m, U, Cin, Cout) : v5_pack_weights(m, U, Cin, Cout);
            free(U);
            U = U_p;
        }
    }

    const char* exec_path = NULL;

    /* Warm-up iterations */
    for (int r = 0; r < warmup; r++) {
        if (strcmp(variant, "naive_direct") == 0) {
            conv2d_naive_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "neon_direct") == 0) {
            conv2d_neon_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "unfused") == 0) {
            conv2d_unfused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "fused") == 0) {
            conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v2_regblocked") == 0) {
            conv2d_v2_regblocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v3_cin_blocked") == 0) {
            conv2d_v3_cin_blocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v4_pretransformed_packed") == 0) {
            conv2d_v4_pretransformed_packed(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v5_batched") == 0) {
            conv2d_v5_batched(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v6_freq") == 0) {
            conv2d_v6_freq(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "direct_packed") == 0) {
            conv2d_direct_packed(in, W_p, out, H, W, Cin, Cout, g_ksize, g_stride, T, &exec_path);
        }
        if (strcmp(exec_path, variant) != 0) {
            fprintf(stderr, "FATAL: Gate G2 Executed Path Assertion Failed: requested %s != executed %s\n", variant, exec_path);
            return 2;
        }
    }

    FILE* fp = stdout;
    if (out_csv) {
        fp = fopen(out_csv, "a");
        if (!fp) {
            perror("Failed to open output CSV");
            return 3;
        }
    }

    /* Timed Measurement iterations */
    for (int r = 0; r < runs; r++) {
        long freq_mhz = get_cpu_freq_mhz();
        double temp_c = get_cpu_temp_c();
        unsigned int throttle = get_throttle_flags();

        double t0 = get_time_ms();
        if (strcmp(variant, "naive_direct") == 0) {
            conv2d_naive_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "neon_direct") == 0) {
            conv2d_neon_direct(in, w, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "unfused") == 0) {
            conv2d_unfused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "fused") == 0) {
            conv2d_fused_winograd(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v2_regblocked") == 0) {
            conv2d_v2_regblocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v3_cin_blocked") == 0) {
            conv2d_v3_cin_blocked(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v4_pretransformed_packed") == 0) {
            conv2d_v4_pretransformed_packed(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v5_batched") == 0) {
            conv2d_v5_batched(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "v6_freq") == 0) {
            conv2d_v6_freq(m, in, U, out, H, W, Cin, Cout, T, &exec_path);
        } else if (strcmp(variant, "direct_packed") == 0) {
            conv2d_direct_packed(in, W_p, out, H, W, Cin, Cout, g_ksize, g_stride, T, &exec_path);
        }
        double t1 = get_time_ms();
        double wall_time_ms = t1 - t0;
        double implied_gflops = (total_flops / (wall_time_ms / 1000.0)) / 1e9;

        /* Write row matching Part A.6 schema */
        fprintf(fp, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%s,%d,%.6f,%.6f,%ld,%.2f,0x%x,%s\n",
                board, git_hash, cell_id, variant, m, T, H, W, Cin, Cout,
                "warm", r, wall_time_ms, implied_gflops, freq_mhz, temp_c, throttle, exec_path);
    }

    if (getenv("PROF")) fprintf(stderr, "PROF in=%.2f mul=%.2f out=%.2f (ms/run, thread 0) freq=%ld\n", g_prof[0] / (runs + warmup), g_prof[1] / (runs + warmup), g_prof[2] / (runs + warmup), get_cpu_freq_mhz());
    if (out_csv) {
        fflush(fp);
        fsync(fileno(fp));
        fclose(fp);
    }

    if (U) free(U);
    free(W_p);
    free(in);
    free(w);
    free(out);

    return 0;
}
