// Direct vs Winograd 3x3 stride-1 convolutions on the GAP8, for ResNet-8's layer shapes.
// float32 runs in software (the GAP8 has no FPU); int8 accumulates in int32.
// Engines: the fabric controller, 1 cluster core and 8 cluster cores. Data stays in L2
// for now; each cluster core's Winograd scratch is in L1. One JSON line per run.
// int8 Winograd uses integer transforms scaled by 2, so it must equal direct exactly.
#include "pmsis.h"

typedef struct { int C, H; } Shape;                 // Cin = Cout = C, output H x H, input (H+2)^2
#ifdef QUICK
static const Shape SHAPES[] = {{8, 8}};               // a tiny layer, to check every path quickly
#else
static const Shape SHAPES[] = {{16, 32}, {32, 16}, {64, 8}};
#endif
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))
enum { FP32, INT8 };
enum { DIRECT, W23, W43 };
static const char *FMT[] = {"fp32", "int8"}, *ALGO[] = {"direct", "wino_f23", "wino_f43"};

static struct {
    int fmt, algo, C, H, nc, on_fc;
    void *in, *w, *u, *out;                          // u: transformed weights (Winograd)
    uint8_t *scratch; int scratch_core;              // per-core V buffers
} job;
static uint32_t cyc;

/* ---------------- 1-D transforms (columns then rows gives the 2-D one) ---------------- */
#define BT2(T, d, s, o, os) do { T d0 = d[0], d1 = d[s], d2 = d[2*(s)], d3 = d[3*(s)]; \
    o[0] = d0 - d2; o[os] = d1 + d2; o[2*(os)] = d2 - d1; o[3*(os)] = d1 - d3; } while (0)
#define AT2(T, m, s, o, os) do { T m1 = m[s], m2 = m[2*(s)]; \
    o[0] = m[0] + m1 + m2; o[os] = m1 - m2 - m[3*(s)]; } while (0)

static inline void bt4(const float *d, int s, float *o, int os) {
    float d0 = d[0], d1 = d[s], d2 = d[2*s], d3 = d[3*s], d4 = d[4*s], d5 = d[5*s];
    float p = d3 + d4, q = d1 + d2, r = d4 - d3, u = d1 - d2, w = d4 - d2, x = d3 - d1;
    o[0] = d0 * 4.0f - d2 * 5.0f + d4;  o[os] = p - q * 4.0f;     o[2*os] = r + u * 4.0f;
    o[3*os] = w + x * 2.0f;              o[4*os] = w - x * 2.0f;   o[5*os] = d1 * 4.0f - d3 * 5.0f + d5;
}
static inline void at4(const float *m, int s, float *o, int os) {
    float m1 = m[s], m2 = m[2*s], m3 = m[3*s], m4 = m[4*s];
    float a = m1 + m2, b = m1 - m2, c = m3 + m4, d = m3 - m4;
    o[0] = m[0] + a + c; o[os] = b + d * 2.0f; o[2*os] = a + c * 4.0f; o[3*os] = b + d * 8.0f + m[5*s];
}

/* ---------------- weight transforms, done once at setup (not timed) ---------------- */
static const float G2[4][3] = {{1, 0, 0}, {.5f, .5f, .5f}, {.5f, -.5f, .5f}, {0, 0, 1}};
static const int   G2I[4][3] = {{2, 0, 0}, {1, 1, 1}, {1, -1, 1}, {0, 0, 2}};    // 2 * G2
static const float G4[6][3] = {{1/4.f, 0, 0}, {-1/6.f, -1/6.f, -1/6.f}, {-1/6.f, 1/6.f, -1/6.f},
                               {1/24.f, 1/12.f, 1/6.f}, {1/24.f, -1/12.f, 1/6.f}, {0, 0, 1}};

static void wt_f32(const float *w, float *u, int C, int a, const float *G) {   // u = G g G^T
    for (int i = 0; i < C * C; i++) {
        const float *g = w + i * 9; float t[18], *o = u + i * a * a;
        for (int r = 0; r < a; r++) for (int c = 0; c < 3; c++)
            t[r*3+c] = G[r*3] * g[c] + G[r*3+1] * g[3+c] + G[r*3+2] * g[6+c];
        for (int r = 0; r < a; r++) for (int c = 0; c < a; c++)
            o[r*a+c] = t[r*3] * G[c*3] + t[r*3+1] * G[c*3+1] + t[r*3+2] * G[c*3+2];
    }
}
static void wt_i8(const int8_t *w, int16_t *u, int C) {                           // u = (2G) g (2G)^T = 4U
    for (int i = 0; i < C * C; i++) {
        const int8_t *g = w + i * 9; int t[12]; int16_t *o = u + i * 16;
        for (int r = 0; r < 4; r++) for (int c = 0; c < 3; c++)
            t[r*3+c] = G2I[r][0] * g[c] + G2I[r][1] * g[3+c] + G2I[r][2] * g[6+c];
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
            o[r*4+c] = t[r*3] * G2I[c][0] + t[r*3+1] * G2I[c][1] + t[r*3+2] * G2I[c][2];
    }
}

/* ---------------- kernels: core `core` of `nc` ---------------- */
static void direct_f32(int core, int nc) {
    const int C = job.C, H = job.H, Wp = H + 2; const float *in = job.in, *w = job.w; float *out = job.out;
    for (int co = core; co < C; co += nc)
        for (int y = 0; y < H; y++) for (int x = 0; x < H; x++) {
            float acc = 0;
            for (int ci = 0; ci < C; ci++) {
                const float *p = in + (ci * Wp + y) * Wp + x, *k = w + (co * C + ci) * 9;
                acc += p[0]*k[0] + p[1]*k[1] + p[2]*k[2] + p[Wp]*k[3] + p[Wp+1]*k[4] + p[Wp+2]*k[5]
                     + p[2*Wp]*k[6] + p[2*Wp+1]*k[7] + p[2*Wp+2]*k[8];
            }
            out[(co * H + y) * H + x] = acc;
        }
}
static void direct_i8(int core, int nc) {
    const int C = job.C, H = job.H, Wp = H + 2; const int8_t *in = job.in, *w = job.w; int32_t *out = job.out;
    for (int co = core; co < C; co += nc)
        for (int y = 0; y < H; y++) for (int x = 0; x < H; x++) {
            int32_t acc = 0;
            for (int ci = 0; ci < C; ci++) {
                const int8_t *p = in + (ci * Wp + y) * Wp + x, *k = w + (co * C + ci) * 9;
                acc += p[0]*k[0] + p[1]*k[1] + p[2]*k[2] + p[Wp]*k[3] + p[Wp+1]*k[4] + p[Wp+2]*k[5]
                     + p[2*Wp]*k[6] + p[2*Wp+1]*k[7] + p[2*Wp+2]*k[8];
            }
            out[(co * H + y) * H + x] = acc;
        }
}
/* Winograd: each core takes whole tiles: input transform for every channel, then every output channel */
static void wino_f32(int core, int nc) {
    const int C = job.C, H = job.H, Wp = H + 2, m = job.algo == W23 ? 2 : 4, a = m + 2, a2 = a * a, Tw = H / m;
    const float *in = job.in, *u = job.u; float *out = job.out;
    float *V = (float *)(job.scratch + core * job.scratch_core);
    for (int t = core; t < Tw * Tw; t += nc) {
        int y0 = t / Tw * m, x0 = t % Tw * m;
        for (int ci = 0; ci < C; ci++) {
            float d[36], s[36]; const float *p = in + (ci * Wp + y0) * Wp + x0;
            for (int r = 0; r < a; r++) for (int c = 0; c < a; c++) d[r*a+c] = p[r*Wp+c];
            float *v = V + ci * a2;
            if (m == 2) { for (int c = 0; c < 4; c++) BT2(float, (d+c), 4, (s+c), 4); for (int r = 0; r < 4; r++) BT2(float, (s+r*4), 1, (v+r*4), 1); }
            else        { for (int c = 0; c < 6; c++) bt4(d+c, 6, s+c, 6);            for (int r = 0; r < 6; r++) bt4(s+r*6, 1, v+r*6, 1); }
        }
        for (int co = 0; co < C; co++) {
            float mm[36] = {0}, s[24], y[16];
            for (int ci = 0; ci < C; ci++) {
                const float *uu = u + (co * C + ci) * a2, *v = V + ci * a2;
                for (int k = 0; k < a2; k++) mm[k] += uu[k] * v[k];
            }
            if (m == 2) { for (int c = 0; c < 4; c++) AT2(float, (mm+c), 4, (s+c), 4); for (int r = 0; r < 2; r++) AT2(float, (s+r*4), 1, (y+r*2), 1); }
            else        { for (int c = 0; c < 6; c++) at4(mm+c, 6, s+c, 6);            for (int r = 0; r < 4; r++) at4(s+r*6, 1, y+r*4, 1); }
            for (int r = 0; r < m; r++) for (int c = 0; c < m; c++) out[(co * H + y0 + r) * H + x0 + c] = y[r*m+c];
        }
    }
}
static void wino23_i8(int core, int nc) {
    const int C = job.C, H = job.H, Wp = H + 2, Tw = H / 2;
    const int8_t *in = job.in; const int16_t *u = job.u; int32_t *out = job.out;
    int16_t *V = (int16_t *)(job.scratch + core * job.scratch_core);
    for (int t = core; t < Tw * Tw; t += nc) {
        int y0 = t / Tw * 2, x0 = t % Tw * 2;
        for (int ci = 0; ci < C; ci++) {
            int d[16], s[16]; const int8_t *p = in + (ci * Wp + y0) * Wp + x0; int16_t *v = V + ci * 16;
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) d[r*4+c] = p[r*Wp+c];
            for (int c = 0; c < 4; c++) BT2(int, (d+c), 4, (s+c), 4);
            for (int r = 0; r < 4; r++) { int o[4]; BT2(int, (s+r*4), 1, o, 1); for (int c = 0; c < 4; c++) v[r*4+c] = o[c]; }
        }
        for (int co = 0; co < C; co++) {
            int32_t mm[16] = {0}, s[8], y[4];
            for (int ci = 0; ci < C; ci++) {
                const int16_t *uu = u + (co * C + ci) * 16, *v = V + ci * 16;
                for (int k = 0; k < 16; k++) mm[k] += uu[k] * v[k];
            }
            for (int c = 0; c < 4; c++) AT2(int32_t, (mm+c), 4, (s+c), 4);
            for (int r = 0; r < 2; r++) AT2(int32_t, (s+r*4), 1, (y+r*2), 1);
            for (int r = 0; r < 2; r++) for (int c = 0; c < 2; c++) out[(co * H + y0 + r) * H + x0 + c] = y[r*2+c] / 4;  // exact
        }
    }
}

static void worker(void *arg) {
    int core = job.on_fc ? 0 : pi_core_id(), nc = job.nc;
    if (job.fmt == FP32) { if (job.algo == DIRECT) direct_f32(core, nc); else wino_f32(core, nc); }
    else                 { if (job.algo == DIRECT) direct_i8(core, nc);  else wino23_i8(core, nc); }
}

static int scratch_bytes(void) {
    if (job.algo == DIRECT) return 0;
    int a = job.algo == W23 ? 4 : 6;
    return (job.C * a * a * (job.fmt == FP32 ? 4 : 2) + 7) & ~7;
}
static void cl_run(void *arg) {
    int n = job.nc * job.scratch_core;
    job.scratch = n ? pi_cl_l1_malloc(NULL, n) : NULL;
    if (n && !job.scratch) { cyc = 0; return; }
    pi_perf_conf(1 << PI_PERF_CYCLES); pi_perf_reset(); pi_perf_start();
    pi_cl_team_fork(job.nc, worker, NULL);
    pi_perf_stop();
    cyc = pi_perf_read(PI_PERF_CYCLES);
    if (n) pi_cl_l1_free(NULL, job.scratch, n);
}

static uint32_t lcg = 12345;
static int rnd(void) { lcg = lcg * 1103515245u + 12345u; return (int)(lcg >> 16) & 0x7fff; }

static void run_one(struct pi_device *cl, const char *eng, void *ref, int n_out) {
    job.scratch_core = scratch_bytes();
    job.on_fc = eng[0] == 'f';
    job.nc = eng[0] == 'f' ? 1 : eng[2] - '0';        // "fc", "cl1", "cl8"
    memset(job.out, 0, n_out * 4);                    // so a run that computes nothing can't pass the check
    if (job.on_fc) {
        int n = job.scratch_core;
        job.scratch = n ? pi_l2_malloc(n) : NULL;
        if (n && !job.scratch) { printf("{\"skip\": \"L2 scratch\"}\n"); return; }
        pi_perf_conf(1 << PI_PERF_CYCLES); pi_perf_reset(); pi_perf_start();
        worker(NULL);
        pi_perf_stop();
        cyc = pi_perf_read(PI_PERF_CYCLES);
        if (n) pi_l2_free(job.scratch, n);
    } else {
        struct pi_cluster_task t;
        pi_cluster_task(&t, cl_run, NULL);
#ifdef BIG_STACKS
        t.stack_size = 2048; t.slave_stack_size = 1536;
#endif
#ifdef QUICK
        printf("  sending %s %s %s\n", FMT[job.fmt], ALGO[job.algo], eng);
#endif
        pi_cluster_send_task_to_cl(cl, &t);
        if (!cyc) { printf("{\"skip\": \"L1 scratch %d B per core\"}\n", job.scratch_core); return; }
    }
    float err = 0, mx = 0; int exact = 1;
    for (int i = 0; i < n_out; i++) {
        if (job.fmt == FP32) {
            float r = ((float *)ref)[i], d = ((float *)job.out)[i] - r;
            if (d < 0) d = -d; if (r < 0) r = -r;
            if (d > err) err = d; if (r > mx) mx = r;
        } else exact &= ((int32_t *)job.out)[i] == ((int32_t *)ref)[i];
    }
    int macs = job.C * job.C * 9 * job.H * job.H;
    printf("{\"fmt\": \"%s\", \"algo\": \"%s\", \"C\": %d, \"H\": %d, \"engine\": \"%s\", \"cycles\": %u, "
           "\"cycles_per_direct_mac\": %d.%02d, ", FMT[job.fmt], ALGO[job.algo], job.C, job.H, eng, cyc,
           (int)(cyc / macs), (int)(cyc % macs * 100 / macs));
    if (job.fmt == FP32) printf("\"max_rel_err_e9\": %d}\n", (int)(err / mx * 1e9f));
    else printf("\"identical_to_direct\": %d}\n", exact);
}

static void bench(void) {
    struct pi_device cl; struct pi_cluster_conf conf;
    pi_cluster_conf_init(&conf); pi_open_from_conf(&cl, &conf);
    if (pi_cluster_open(&cl)) { printf("cluster open failed\n"); pmsis_exit(-1); }
    printf("{\"board\": \"GAP8\", \"fc_hz\": %d, \"cl_hz\": %d}\n", pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL));
    static const char *ENG[] = {"fc", "cl1", "cl8"};
    for (int s = 0; s < N_SHAPES; s++) for (int f = 0; f < 2; f++) {
        int C = SHAPES[s].C, H = SHAPES[s].H, Wp = H + 2, e = f == FP32 ? 4 : 1;
        int nin = C * Wp * Wp, nw = C * C * 9, nout = C * H * H;
        job.fmt = f; job.C = C; job.H = H;
        job.in = pi_l2_malloc(nin * e); job.w = pi_l2_malloc(nw * e);
        void *ref = pi_l2_malloc(nout * 4), *out = pi_l2_malloc(nout * 4);
        if (!job.in || !job.w || !out || !ref) { printf("{\"skip\": \"L2 data C=%d\"}\n", C); pmsis_exit(-2); }
        for (int i = 0; i < nin; i++) {                       // zero border = padding 1
            int c = i % Wp, r = i / Wp % Wp, v = (r == 0 || c == 0 || r == Wp - 1 || c == Wp - 1) ? 0 : rnd() % 256 - 128;
            if (f == FP32) ((float *)job.in)[i] = v / 128.0f; else ((int8_t *)job.in)[i] = v;
        }
        for (int i = 0; i < nw; i++) {
            int v = rnd() % 256 - 128;
            if (f == FP32) ((float *)job.w)[i] = v / 256.0f; else ((int8_t *)job.w)[i] = v;
        }
        job.algo = DIRECT; job.out = ref; job.on_fc = 1; job.nc = 1; worker(NULL);   // reference
        job.out = out;
        for (int g = 0; g < 3; g++) run_one(&cl, ENG[g], ref, nout);
        int algos[] = {W23, W43};
        for (int ai = 0; ai < (f == FP32 ? 2 : 1); ai++) {
            int a = algos[ai] == W23 ? 4 : 6, nu = C * C * a * a * (f == FP32 ? 4 : 2);
            job.algo = algos[ai];
            job.u = pi_l2_malloc(nu);
            if (!job.u) { printf("{\"fmt\": \"%s\", \"algo\": \"%s\", \"C\": %d, \"skip\": \"transformed weights %d B do not fit L2\"}\n", FMT[f], ALGO[job.algo], C, nu); continue; }
            if (f == FP32) wt_f32(job.w, job.u, C, a, a == 4 ? &G2[0][0] : &G4[0][0]); else wt_i8(job.w, job.u, C);
            for (int g = 0; g < 3; g++) run_one(&cl, ENG[g], ref, nout);
            pi_l2_free(job.u, nu);
        }
        pi_l2_free(job.in, nin * e); pi_l2_free(job.w, nw * e); pi_l2_free(job.out, nout * 4); pi_l2_free(ref, nout * 4);
    }
    pi_cluster_close(&cl);
    printf("{\"done\": 1}\n");
    pmsis_exit(0);
}

int main(void) { return pmsis_kickoff((void *)bench); }
