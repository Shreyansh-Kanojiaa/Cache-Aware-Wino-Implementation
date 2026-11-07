/*
 * src/model_runner.c
 *
 * Runs a whole model in C from a plan made by benchmarks/whole_model.py, using the
 * same graph and weights as the ONNX file. 3x3 stride-1 convs use --variant, depthwise
 * convs use conv2d_depthwise, and every other conv uses direct_packed.
 */
#define main harness_main
#include "bench_harness.c"
#undef main

enum { OP_CONV, OP_RELU, OP_MAXPOOL, OP_ADD, OP_GAP, OP_GEMM, OP_CLIP, OP_SILU, OP_CONCAT, OP_UPSAMPLE };
enum { CAT_CONV3, CAT_CONV_OTHER, CAT_PAD, CAT_RELU, CAT_POOL, CAT_ADD_GAP, CAT_GEMM,
       CAT_DW, CAT_ACT, CAT_CONCAT, N_CATS };
static const char* CAT_NAMES[N_CATS] = {
    "conv3x3_s1_ms", "conv_other_ms", "pad_ms", "relu_ms", "pool_ms", "add_gap_ms", "gemm_ms",
    "conv_depthwise_ms", "clip_silu_ms", "concat_upsample_ms"
};

typedef struct { int c, h, w; float* d; } Tensor;
typedef struct {
    int type, out, in, in2;
    int cin, cout, k, s, p, m, use_v5, depthwise;
    float lo, hi;       /* clip */
    int n_ins, ins[8];  /* concat */
    int nin, nout;
    long woff, boff;
    float* pack;        /* packed conv weights */
    float* pad;         /* zero-padded input, conv only */
    const float* w;     /* gemm weights */
    const float* bias;
} Op;

#define MAX_TENSORS 256
#define MAX_OPS 256
static Tensor g_t[MAX_TENSORS];
static Op g_ops[MAX_OPS];
static int g_nops;
static double g_cat[N_CATS];
#define MAX_OUTPUTS 8
static int g_outs[MAX_OUTPUTS];
static int g_n_outs;

static void die(const char* msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(2);
}

/* x / (1 + exp(-x)), using the Cephes float exp polynomial */
static inline void silu_plane(const float* x, float* y, size_t n) {
    size_t i = 0;
#if HAS_ARM_NEON
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t a = vminq_f32(vmaxq_f32(vnegq_f32(v), vdupq_n_f32(-88.3762626647949f)),
                                  vdupq_n_f32(88.3762626647949f));
        int32x4_t e = vcvtnq_s32_f32(vmulq_n_f32(a, 1.44269504088896341f));
        float32x4_t ef = vcvtq_f32_s32(e);
        float32x4_t r = vfmsq_n_f32(a, ef, 0.693359375f);
        r = vfmsq_n_f32(r, ef, -2.12194440e-4f);
        float32x4_t p = vdupq_n_f32(1.9875691500E-4f);
        p = vfmaq_f32(vdupq_n_f32(1.3981999507E-3f), p, r);
        p = vfmaq_f32(vdupq_n_f32(8.3334519073E-3f), p, r);
        p = vfmaq_f32(vdupq_n_f32(4.1665795894E-2f), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.6666665459E-1f), p, r);
        p = vfmaq_f32(vdupq_n_f32(5.0000001201E-1f), p, r);
        p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), vmulq_f32(p, r), r);
        float32x4_t scale = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23));
        vst1q_f32(y + i, vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.0f), vmulq_f32(p, scale))));
    }
#endif
    for (; i < n; i++) y[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ReLU and ReLU6. In-place is fine. Threads split fixed blocks, so flat tensors work too. */
static void clip_tensor(const float* x, float* y, size_t n, float lo, float hi, int T) {
    enum { BLOCK = 4096 };
    long blocks = (long)((n + BLOCK - 1) / BLOCK);
    #pragma omp parallel for num_threads(T) schedule(static)
    for (long b = 0; b < blocks; b++) {
        size_t i = (size_t)b * BLOCK;
        size_t end = i + BLOCK < n ? i + BLOCK : n;
#if HAS_ARM_NEON
        float32x4_t vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
        for (; i + 4 <= end; i += 4) vst1q_f32(y + i, vminq_f32(vmaxq_f32(vld1q_f32(x + i), vlo), vhi));
#endif
        for (; i < end; i++) y[i] = x[i] < lo ? lo : (x[i] > hi ? hi : x[i]);
    }
}

static void run_conv(Op* o, int T) {
    Tensor* in = &g_t[o->in];
    Tensor* out = &g_t[o->out];
    const float* src = in->d;
    int H = in->h, W = in->w;
    const char* path = NULL;

    double t0 = get_time_ms();
    if (o->p) {
        int Hp = H + 2 * o->p, Wp = W + 2 * o->p;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int ci = 0; ci < o->cin; ci++) {
            for (int r = 0; r < H; r++) {
                memcpy(o->pad + ((size_t)ci * Hp + r + o->p) * Wp + o->p,
                       src + ((size_t)ci * H + r) * W, (size_t)W * sizeof(float));
            }
        }
        src = o->pad;
        H = Hp;
        W = Wp;
    }
    double t1 = get_time_ms();
    g_cat[CAT_PAD] += t1 - t0;

    if (o->depthwise) {
        conv2d_depthwise(src, o->w, out->d, H, W, o->cin, o->k, o->s, T);
    } else if (o->use_v5) {
        conv2d_v5_batched(o->m, src, o->pack, out->d, H, W, o->cin, o->cout, T, &path);
    } else {
        conv2d_direct_packed(src, o->pack, out->d, H, W, o->cin, o->cout, o->k, o->s, T, &path);
    }
    if (o->bias) {
        size_t plane = (size_t)out->h * out->w;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int co = 0; co < o->cout; co++) {
            float* p = out->d + co * plane;
            float b = o->bias[co];
            for (size_t i = 0; i < plane; i++) p[i] += b;
        }
    }
    g_cat[o->depthwise ? CAT_DW : (o->k == 3 && o->s == 1) ? CAT_CONV3 : CAT_CONV_OTHER] += get_time_ms() - t1;
}

static void run_op(Op* o, int T) {
    if (o->type == OP_CONV) {
        run_conv(o, T);
        return;
    }
    double t0 = get_time_ms();
    Tensor* out = &g_t[o->out];
    Tensor* in = &g_t[o->in];
    size_t n = (size_t)out->c * out->h * out->w;
    int cat = CAT_ADD_GAP;

    if (o->type == OP_RELU) {
        cat = CAT_RELU;
        clip_tensor(in->d, out->d, n, 0.0f, INFINITY, T);
    } else if (o->type == OP_CLIP) {
        cat = CAT_ACT;
        clip_tensor(in->d, out->d, n, o->lo, o->hi, T);
    } else if (o->type == OP_SILU) {
        cat = CAT_ACT;
        size_t plane = (size_t)in->h * in->w;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int c = 0; c < in->c; c++) silu_plane(in->d + c * plane, out->d + c * plane, plane);
    } else if (o->type == OP_CONCAT) {
        cat = CAT_CONCAT;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int i = 0; i < o->n_ins; i++) {
            size_t before = 0;
            for (int j = 0; j < i; j++) before += g_t[o->ins[j]].c;
            Tensor* t = &g_t[o->ins[i]];
            memcpy(out->d + before * t->h * t->w, t->d, (size_t)t->c * t->h * t->w * sizeof(float));
        }
    } else if (o->type == OP_UPSAMPLE) {
        cat = CAT_CONCAT;
        int H = in->h, W = in->w;
        #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
        for (int c = 0; c < in->c; c++) {
            for (int r = 0; r < H; r++) {
                const float* p = in->d + ((size_t)c * H + r) * W;
                float* y = out->d + ((size_t)c * (H * 2) + r * 2) * (W * 2);
                for (int x = 0; x < W; x++) y[x * 2] = y[x * 2 + 1] = p[x];
                memcpy(y + W * 2, y, (size_t)W * 2 * sizeof(float));
            }
        }
    } else if (o->type == OP_ADD) {
        const float* a = in->d;
        const float* b = g_t[o->in2].d;
        float* y = out->d;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (size_t i = 0; i < n; i++) y[i] = a[i] + b[i];
    } else if (o->type == OP_GAP) {
        size_t plane = (size_t)in->h * in->w;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int c = 0; c < in->c; c++) {
            const float* p = in->d + c * plane;
            double sum = 0.0;
            for (size_t i = 0; i < plane; i++) sum += p[i];
            out->d[c] = (float)(sum / (double)plane);
        }
    } else if (o->type == OP_MAXPOOL) {
        cat = CAT_POOL;
        int Ho = out->h, Wo = out->w, H = in->h, W = in->w;
        #pragma omp parallel for collapse(2) num_threads(T) schedule(static)
        for (int c = 0; c < in->c; c++) {
            for (int ho = 0; ho < Ho; ho++) {
                const float* p = in->d + (size_t)c * H * W;
                float* y = out->d + ((size_t)c * Ho + ho) * Wo;
                for (int wo = 0; wo < Wo; wo++) {
                    int r0 = ho * o->s - o->p, c0 = wo * o->s - o->p;
                    int r1 = r0 + o->k < H ? r0 + o->k : H;
                    int c1 = c0 + o->k < W ? c0 + o->k : W;
                    float best = -INFINITY;
                    for (int r = r0 < 0 ? 0 : r0; r < r1; r++) {
                        for (int cc = c0 < 0 ? 0 : c0; cc < c1; cc++) {
                            if (p[r * W + cc] > best) best = p[r * W + cc];
                        }
                    }
                    y[wo] = best;
                }
            }
        }
    } else if (o->type == OP_GEMM) {
        cat = CAT_GEMM;
        const float* x = in->d;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int row = 0; row < o->nout; row++) {
            const float* w = o->w + (size_t)row * o->nin;
            v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
            int j = 0;
            for (; j + 4 <= o->nin; j += 4) acc += *(const v4f*)(w + j) * *(const v4f*)(x + j);
            float sum = acc[0] + acc[1] + acc[2] + acc[3];
            for (; j < o->nin; j++) sum += w[j] * x[j];
            out->d[row] = sum + (o->bias ? o->bias[row] : 0.0f);
        }
    }
    g_cat[cat] += get_time_ms() - t0;
}

static void load_plan(const char* path, const float* weights, int use_v5, int* in_id) {
    FILE* f = fopen(path, "r");
    if (!f) die("cannot open plan");
    char word[32];
    while (fscanf(f, "%31s", word) == 1) {
        Op* o = &g_ops[g_nops];
        memset(o, 0, sizeof(*o));
        if (strcmp(word, "TENSOR") == 0) {
            int id, c, h, w;
            if (fscanf(f, "%d %d %d %d", &id, &c, &h, &w) != 4 || id >= MAX_TENSORS) die("bad TENSOR");
            g_t[id] = (Tensor){c, h, w, v5_alloc((size_t)c * h * w)};
            continue;
        }
        if (strcmp(word, "INPUT") == 0) {
            if (fscanf(f, "%d", in_id) != 1) die("bad INPUT");
            continue;
        }
        if (strcmp(word, "OUTPUT") == 0) {
            if (g_n_outs >= MAX_OUTPUTS || fscanf(f, "%d", &g_outs[g_n_outs++]) != 1) die("bad OUTPUT");
            continue;
        }
        if (strcmp(word, "SPLIT") == 0) {
            /* Channels are contiguous, so both halves point into the input: no op, no copy */
            int a, b, src;
            if (fscanf(f, "%d %d %d", &a, &b, &src) != 3) die("bad SPLIT");
            free(g_t[a].d);
            free(g_t[b].d);
            g_t[a].d = g_t[src].d;
            g_t[b].d = g_t[src].d + (size_t)g_t[a].c * g_t[a].h * g_t[a].w;
            continue;
        }
        if (strcmp(word, "CONV") == 0) {
            o->type = OP_CONV;
            if (fscanf(f, "%d %d %d %d %d %d %d %d %ld %ld", &o->out, &o->in, &o->cin, &o->cout,
                       &o->k, &o->s, &o->p, &o->m, &o->woff, &o->boff) != 10) die("bad CONV");
            o->use_v5 = use_v5 && o->k == 3 && o->s == 1;
            if (o->use_v5) {
                int alpha = o->m + 2;
                float* U = (float*)malloc((size_t)o->cout * o->cin * alpha * alpha * sizeof(float));
                transform_weights(o->m, o->cin, o->cout, weights + o->woff, U);
                o->pack = v5_pack_weights(o->m, U, o->cin, o->cout);
                free(U);
            } else {
                o->pack = dp_pack_weights(weights + o->woff, o->cin, o->cout, o->k);
            }
            o->bias = o->boff >= 0 ? weights + o->boff : NULL;
            if (o->p) {
                Tensor* in = &g_t[o->in];
                size_t n = (size_t)o->cin * (in->h + 2 * o->p) * (in->w + 2 * o->p);
                o->pad = v5_alloc(n);
                memset(o->pad, 0, n * sizeof(float));
            }
        } else if (strcmp(word, "DWCONV") == 0) {
            o->type = OP_CONV;
            o->depthwise = 1;
            if (fscanf(f, "%d %d %d %d %d %d %ld %ld", &o->out, &o->in, &o->cin, &o->k, &o->s, &o->p,
                       &o->woff, &o->boff) != 8) die("bad DWCONV");
            o->cout = o->cin;
            o->w = weights + o->woff;
            o->bias = o->boff >= 0 ? weights + o->boff : NULL;
            if (o->p) {
                Tensor* in = &g_t[o->in];
                size_t n = (size_t)o->cin * (in->h + 2 * o->p) * (in->w + 2 * o->p);
                o->pad = v5_alloc(n);
                memset(o->pad, 0, n * sizeof(float));
            }
        } else if (strcmp(word, "CLIP") == 0) {
            o->type = OP_CLIP;
            if (fscanf(f, "%d %d %f %f", &o->out, &o->in, &o->lo, &o->hi) != 4) die("bad CLIP");
        } else if (strcmp(word, "SILU") == 0) {
            o->type = OP_SILU;
            if (fscanf(f, "%d %d", &o->out, &o->in) != 2) die("bad SILU");
        } else if (strcmp(word, "UPSAMPLE") == 0) {
            o->type = OP_UPSAMPLE;
            if (fscanf(f, "%d %d", &o->out, &o->in) != 2) die("bad UPSAMPLE");
        } else if (strcmp(word, "CONCAT") == 0) {
            o->type = OP_CONCAT;
            if (fscanf(f, "%d %d", &o->out, &o->n_ins) != 2 || o->n_ins > 8) die("bad CONCAT");
            for (int i = 0; i < o->n_ins; i++) {
                if (fscanf(f, "%d", &o->ins[i]) != 1) die("bad CONCAT");
            }
            o->in = o->ins[0];
        } else if (strcmp(word, "RELU") == 0) {
            o->type = OP_RELU;
            if (fscanf(f, "%d %d", &o->out, &o->in) != 2) die("bad RELU");
        } else if (strcmp(word, "MAXPOOL") == 0) {
            o->type = OP_MAXPOOL;
            if (fscanf(f, "%d %d %d %d %d", &o->out, &o->in, &o->k, &o->s, &o->p) != 5) die("bad MAXPOOL");
        } else if (strcmp(word, "ADD") == 0) {
            o->type = OP_ADD;
            if (fscanf(f, "%d %d %d", &o->out, &o->in, &o->in2) != 3) die("bad ADD");
        } else if (strcmp(word, "GAP") == 0) {
            o->type = OP_GAP;
            if (fscanf(f, "%d %d", &o->out, &o->in) != 2) die("bad GAP");
        } else if (strcmp(word, "GEMM") == 0) {
            o->type = OP_GEMM;
            if (fscanf(f, "%d %d %d %d %ld %ld", &o->out, &o->in, &o->nin, &o->nout, &o->woff, &o->boff) != 6) die("bad GEMM");
            o->w = weights + o->woff;
            o->bias = o->boff >= 0 ? weights + o->boff : NULL;
        } else {
            die("unknown plan op");
        }
        if (++g_nops >= MAX_OPS) die("too many ops");
    }
    fclose(f);
}

static float* read_floats(const char* path, size_t* count) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open float file");
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    float* d = v5_alloc(bytes / sizeof(float));
    if (fread(d, 1, bytes, f) != bytes) die("short read");
    fclose(f);
    if (count) *count = bytes / sizeof(float);
    return d;
}

int main(int argc, char** argv) {
    const char* plan = NULL, *weights_path = NULL, *input_path = NULL, *output_path = NULL, *csv = NULL;
    const char* variant = "v5_batched";
    int T = 4, warmup = 10, runs = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--plan") == 0) plan = argv[i + 1];
        else if (strcmp(argv[i], "--weights") == 0) weights_path = argv[i + 1];
        else if (strcmp(argv[i], "--input") == 0) input_path = argv[i + 1];
        else if (strcmp(argv[i], "--output") == 0) output_path = argv[i + 1];
        else if (strcmp(argv[i], "--csv") == 0) csv = argv[i + 1];
        else if (strcmp(argv[i], "--variant") == 0) variant = argv[i + 1];
        else if (strcmp(argv[i], "--T") == 0) T = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--warmup") == 0) warmup = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--runs") == 0) runs = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--batch") == 0) g_v5_batch = atoi(argv[i + 1]);
        else die("unknown argument");
    }
    if (!plan || !weights_path || !input_path || !output_path || !csv) die("missing argument");
    if (strcmp(variant, "v5_batched") != 0 && strcmp(variant, "direct_packed") != 0) die("unknown variant");

    float* weights = read_floats(weights_path, NULL);
    int in_id = -1;
    load_plan(plan, weights, strcmp(variant, "v5_batched") == 0, &in_id);
    if (in_id < 0 || g_n_outs == 0) die("plan has no INPUT or OUTPUT");

    size_t n_in = 0;
    float* input = read_floats(input_path, &n_in);
    if (n_in != (size_t)g_t[in_id].c * g_t[in_id].h * g_t[in_id].w) die("input size mismatch");
    memcpy(g_t[in_id].d, input, n_in * sizeof(float));

    for (int r = 0; r < warmup; r++) {
        for (int i = 0; i < g_nops; i++) run_op(&g_ops[i], T);
    }

    FILE* fp = fopen(csv, "w");
    if (!fp) die("cannot open csv");
    fprintf(fp, "run_id,total_ms");
    for (int c = 0; c < N_CATS; c++) fprintf(fp, ",%s", CAT_NAMES[c]);
    fprintf(fp, ",freq_mhz,temp_c,throttle_hex\n");
    for (int r = 0; r < runs; r++) {
        long freq = get_cpu_freq_mhz();
        double temp = get_cpu_temp_c();
        unsigned int throttle = get_throttle_flags();
        memset(g_cat, 0, sizeof(g_cat));
        double t0 = get_time_ms();
        for (int i = 0; i < g_nops; i++) run_op(&g_ops[i], T);
        double total = get_time_ms() - t0;
        fprintf(fp, "%d,%.6f", r, total);
        for (int c = 0; c < N_CATS; c++) fprintf(fp, ",%.6f", g_cat[c]);
        fprintf(fp, ",%ld,%.2f,0x%x\n", freq, temp, throttle);
    }
    fclose(fp);

    FILE* fo = fopen(output_path, "wb");
    if (!fo) die("cannot open output");
    for (int i = 0; i < g_n_outs; i++) {
        Tensor* o = &g_t[g_outs[i]];
        fwrite(o->d, sizeof(float), (size_t)o->c * o->h * o->w, fo);
    }
    fclose(fo);
    return 0;
}
