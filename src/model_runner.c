/*
 * src/model_runner.c
 *
 * Runs a whole model in C from a plan made by benchmarks/whole_model.py, using the
 * same graph and weights as the ONNX file. 3x3 stride-1 convs use --variant, depthwise
 * convs use conv2d_depthwise, 1x1 stride-1 convs conv2d_pointwise, and every other conv
 * uses direct_packed.
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
    float lo, hi;       /* clip, or the clamp fused into a conv */
    int act;            /* conv only: 0 none, 1 clamp to [lo, hi], 2 SiLU */
    int n_ins, ins[8];  /* concat */
    int nin, nout;
    long woff, boff;
    float* pack;        /* packed conv weights */
    float* pad;         /* zero-padded input, conv only */
    int pad_filled;     /* the conv before this one writes straight into pad */
    float* out_pad;     /* write the output into the next conv's pad, with an out_p border */
    int out_p;
    const float* w;     /* gemm weights */
    const float* bias;
    float* pw;          /* depthwise only: a fused 1x1 expansion before it (see fuse_pw_dw) */
    const float* pw_bias;
    float pw_lo, pw_hi;
    int pw_cin;
} Op;

#define MAX_TENSORS 256
#define MAX_OPS 256
static Tensor g_t[MAX_TENSORS];
static Op g_ops[MAX_OPS];
static int g_nops;
static double g_cat[N_CATS];
static int g_v6;
static int g_plan_m;   /* 1: v6 uses the plan's tile size instead of v6_pick_m */
static int g_fc_fp16;  /* store fully connected weights as 16-bit floats, as ncnn does by default on Arm */
#define MAX_OUTPUTS 8
static int g_outs[MAX_OUTPUTS];
static int g_n_outs;
static int g_recycle = 1;
static int g_fuse_pwdw = 784;          /* fuse 1x1 + depthwise pairs with at least this many pixels, 0 never */
static int g_fuse_act = 1;             /* apply ReLU, ReLU6 and SiLU in the conv's own output pass */              /* share activation buffers between tensors whose lifetimes don't overlap */
static int g_alias[MAX_TENSORS];       /* SPLIT halves: source tensor + 1, 0 if none */
static size_t g_alias_off[MAX_TENSORS];

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

/* Fully connected: rows are regrouped at load so 8 outputs share one weight stream
 * W_g[group][in][8], and each step reuses one input value across 8 outputs. */
static void fc_pack_rows(float* W, int nin, int nout) {
    float* tmp = (float*)malloc((size_t)8 * nin * sizeof(float));
    if (!tmp) die("fc pack allocation failed");
    for (int g = 0; g < nout / 8; g++) {
        float* blk = W + (size_t)g * 8 * nin;  /* the group's 8 rows are already contiguous */
        memcpy(tmp, blk, (size_t)8 * nin * sizeof(float));
        for (int r = 0; r < 8; r++) {
            for (int k = 0; k < nin; k++) blk[(size_t)k * 8 + r] = tmp[(size_t)r * nin + k];
        }
    }
    free(tmp);
}

static inline void fc_group8(const float* restrict Wg, const float* restrict x, int nin, float* restrict y) {
    v4f z = {0.0f, 0.0f, 0.0f, 0.0f};
    v4f a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    int k = 0;
    for (; k + 4 <= nin; k += 4) {
        v4f xv = *(const v4f*)(x + k);
        a0 += *(const v4f*)(Wg) * xv[0];
        a1 += *(const v4f*)(Wg + 4) * xv[0];
        a2 += *(const v4f*)(Wg + 8) * xv[1];
        a3 += *(const v4f*)(Wg + 12) * xv[1];
        a4 += *(const v4f*)(Wg + 16) * xv[2];
        a5 += *(const v4f*)(Wg + 20) * xv[2];
        a6 += *(const v4f*)(Wg + 24) * xv[3];
        a7 += *(const v4f*)(Wg + 28) * xv[3];
        Wg += 32;
    }
    for (; k < nin; k++) {
        a0 += *(const v4f*)(Wg) * x[k];
        a1 += *(const v4f*)(Wg + 4) * x[k];
        Wg += 8;
    }
    *(v4f*)(y) = (a0 + a2) + (a4 + a6);
    *(v4f*)(y + 4) = (a1 + a3) + (a5 + a7);
}

#if HAS_ARM_NEON
/* Same as fc_group8 with 16-bit weights, widened to 32-bit before multiplying */
static inline void fc_group8_fp16(const __fp16* restrict Wg, const float* restrict x, int nin, float* restrict y) {
    float32x4_t z = vdupq_n_f32(0.0f);
    float32x4_t a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z;
    int k = 0;
    for (; k + 4 <= nin; k += 4) {
        float32x4_t xv = vld1q_f32(x + k);
        float16x8_t h0 = vld1q_f16(Wg), h1 = vld1q_f16(Wg + 8), h2 = vld1q_f16(Wg + 16), h3 = vld1q_f16(Wg + 24);
        a0 = vfmaq_laneq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), xv, 0);
        a1 = vfmaq_laneq_f32(a1, vcvt_high_f32_f16(h0), xv, 0);
        a2 = vfmaq_laneq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), xv, 1);
        a3 = vfmaq_laneq_f32(a3, vcvt_high_f32_f16(h1), xv, 1);
        a4 = vfmaq_laneq_f32(a4, vcvt_f32_f16(vget_low_f16(h2)), xv, 2);
        a5 = vfmaq_laneq_f32(a5, vcvt_high_f32_f16(h2), xv, 2);
        a6 = vfmaq_laneq_f32(a6, vcvt_f32_f16(vget_low_f16(h3)), xv, 3);
        a7 = vfmaq_laneq_f32(a7, vcvt_high_f32_f16(h3), xv, 3);
        Wg += 32;
    }
    for (; k < nin; k++) {
        float16x8_t h = vld1q_f16(Wg);
        a0 = vfmaq_n_f32(a0, vcvt_f32_f16(vget_low_f16(h)), x[k]);
        a1 = vfmaq_n_f32(a1, vcvt_high_f32_f16(h), x[k]);
        Wg += 8;
    }
    vst1q_f32(y, vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a4, a6)));
    vst1q_f32(y + 4, vaddq_f32(vaddq_f32(a1, a3), vaddq_f32(a5, a7)));
}
#endif

/* Bias and activation in one pass over a conv's output */
static void conv_epilogue(Tensor* out, const float* bias, int act, float lo, float hi, int T) {
    size_t plane = (size_t)out->h * out->w;
    #pragma omp parallel for num_threads(T) schedule(static)
    for (int c = 0; c < out->c; c++) {
        float* p = out->d + c * plane;
        float b = bias ? bias[c] : 0.0f;
        if (act == 2) {
            if (bias) for (size_t i = 0; i < plane; i++) p[i] += b;
            silu_plane(p, p, plane);
            continue;
        }
        size_t i = 0;
#if HAS_ARM_NEON
        float32x4_t vb = vdupq_n_f32(b), vlo = vdupq_n_f32(lo), vhi = vdupq_n_f32(hi);
        for (; i + 4 <= plane; i += 4) vst1q_f32(p + i, vminq_f32(vmaxq_f32(vaddq_f32(vld1q_f32(p + i), vb), vlo), vhi));
#endif
        for (; i < plane; i++) {
            float v = p[i] + b;
            p[i] = v < lo ? lo : (v > hi ? hi : v);
        }
    }
}

static void run_conv(Op* o, int T) {
    Tensor* in = &g_t[o->in];
    Tensor* out = &g_t[o->out];
    const float* src = in->d;
    int H = in->h, W = in->w;
    const char* path = NULL;

    double t0 = get_time_ms();
    if (o->p && !o->depthwise) {
        int Hp = H + 2 * o->p, Wp = W + 2 * o->p;
        if (!o->pad_filled) {
            #pragma omp parallel for num_threads(T) schedule(static)
            for (int ci = 0; ci < o->cin; ci++) {
                for (int r = 0; r < H; r++) {
                    memcpy(o->pad + ((size_t)ci * Hp + r + o->p) * Wp + o->p,
                           src + ((size_t)ci * H + r) * W, (size_t)W * sizeof(float));
                }
            }
        }
        src = o->pad;
        H = Hp;
        W = Wp;
    }
    double t1 = get_time_ms();
    g_cat[CAT_PAD] += t1 - t0;

    /* Depthwise and direct_packed add the bias and clamp as they write; anything else
     * they can't do is left for one pass afterwards. */
    float lo = o->act == 1 ? o->lo : -INFINITY, hi = o->act == 1 ? o->hi : INFINITY;
    int done = 0;
    if (o->pw) {
        float plo = o->pw_lo, phi = o->pw_hi;
        conv2d_pw_dw(src, o->pw, o->pw_bias, plo, phi, o->w, o->bias, lo, hi, out->d, H, W, o->pw_cin, o->cin, o->s, T);
        done = 1;
    } else if (o->depthwise) {
        conv2d_depthwise(src, o->w, o->bias, out->d, H, W, o->cin, o->k, o->s, o->p, lo, hi, T);
        done = 1;
    } else if (o->k == 1 && o->s == 1 && o->p == 0) {
        conv2d_pointwise(src, o->pack, o->bias, lo, hi, out->d, H * W, o->cin, o->cout, T);
        done = 1;
    } else if (o->use_v5) {
        if (g_v6) {
            conv2d_v6_freq_ep(o->m, src, o->pack, o->bias, lo, hi, o->out_pad ? o->out_pad : out->d, o->out_p,
                              H, W, o->cin, o->cout, T, &path);
            done = 1;
        } else {
            conv2d_v5_batched(o->m, src, o->pack, out->d, H, W, o->cin, o->cout, T, &path);
        }
    } else {
        conv2d_direct_packed_ep(src, o->pack, o->bias, lo, hi, out->d, H, W, o->cin, o->cout, o->k, o->s, T, &path);
        done = 1;
    }
    if (!done && (o->bias || o->act)) conv_epilogue(out, o->bias, o->act, lo, hi, T);
    else if (done && o->act == 2) conv_epilogue(out, NULL, 2, lo, hi, T);
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
#if HAS_ARM_NEON
                    /* stride 2, k 2 or 3, 4 windows inside the row: deinterleaving loads give
                     * the even and odd columns of all four */
                    if (o->s == 2 && (o->k == 2 || o->k == 3) && c0 >= 0 && wo + 4 <= Wo &&
                        c0 + (o->k == 3 ? 10 : 8) <= W) {
                        float32x4_t best = vdupq_n_f32(-INFINITY);
                        for (int r = r0 < 0 ? 0 : r0; r < r1; r++) {
                            float32x4x2_t a = vld2q_f32(p + (size_t)r * W + c0);
                            best = vmaxq_f32(best, vmaxq_f32(a.val[0], a.val[1]));
                            if (o->k == 3) best = vmaxq_f32(best, vld2q_f32(p + (size_t)r * W + c0 + 2).val[0]);
                        }
                        vst1q_f32(y + wo, best);
                        wo += 3;
                        continue;
                    }
#endif
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
        int groups = o->nout / 8;
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int g = 0; g < groups; g++) {
            float* y = out->d + g * 8;
#if HAS_ARM_NEON
            if (g_fc_fp16) fc_group8_fp16((const __fp16*)o->w + (size_t)g * 8 * o->nin, x, o->nin, y);
            else
#endif
            fc_group8(o->w + (size_t)g * 8 * o->nin, x, o->nin, y);
            for (int r = 0; r < 8; r++) y[r] += o->bias ? o->bias[g * 8 + r] : 0.0f;
        }
        /* rows past the last full group stay row-major */
        for (int row = groups * 8; row < o->nout; row++) {
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

/* Zero-padded input buffers. Convs with the same padded shape share one: each run
 * overwrites only the interior, so the zero border stays valid. */
static float* pad_buffer(int c, int h, int w, int p) {
    static struct { int c, h, w, p; float* d; } bufs[MAX_OPS];
    static int n;
    for (int i = 0; i < n; i++) {
        if (g_recycle && bufs[i].c == c && bufs[i].h == h && bufs[i].w == w && bufs[i].p == p) return bufs[i].d;
    }
    size_t len = (size_t)c * (h + 2 * p) * (w + 2 * p);
    float* d = v5_alloc(len);
    memset(d, 0, len * sizeof(float));
    bufs[n++] = (typeof(bufs[0])){c, h, w, p, d};
    return d;
}

static int root_of(int id) {
    while (g_alias[id]) id = g_alias[id] - 1;
    return id;
}

/* Give every tensor its memory. With recycling, a tensor takes a free buffer as soon as
 * the last op reading its previous owner has run (greedy best fit). Elementwise ops write
 * in place in the plan already. Input and outputs keep their buffers for the whole run,
 * because the input is written once and reused by every run. Returns the bytes used. */
static size_t assign_buffers(int in_id) {
    enum { FOREVER = MAX_OPS + 1 };
    int last[MAX_TENSORS], slot_of[MAX_TENSORS], n_slots = 0;
    size_t cap[MAX_TENSORS];
    int slot_free[MAX_TENSORS];
    for (int t = 0; t < MAX_TENSORS; t++) last[t] = -1, slot_of[t] = -1;
    for (int i = 0; i < g_nops; i++) {
        Op* o = &g_ops[i];
        int ins[10], k = 0;
        if (o->type == OP_CONCAT) for (int j = 0; j < o->n_ins; j++) ins[k++] = o->ins[j];
        else ins[k++] = o->in;
        if (o->type == OP_ADD) ins[k++] = o->in2;
        for (int j = 0; j < k; j++) last[root_of(ins[j])] = i;
        int r = root_of(o->out);
        if (last[r] < i) last[r] = i;  /* written but never read: still needs memory */
    }
    last[root_of(in_id)] = FOREVER;
    for (int j = 0; j < g_n_outs; j++) last[root_of(g_outs[j])] = FOREVER;

    for (int i = -1; i < g_nops; i++) {
        int r = i < 0 ? root_of(in_id) : root_of(g_ops[i].out);
        if (slot_of[r] < 0) {
            size_t need = (size_t)g_t[r].c * g_t[r].h * g_t[r].w;
            int best = -1;
            if (g_recycle) {
                /* smallest free slot that fits, otherwise the largest free one (it grows) */
                for (int s = 0; s < n_slots; s++) {
                    if (!slot_free[s]) continue;
                    if (best < 0) { best = s; continue; }
                    int fits = cap[s] >= need, best_fits = cap[best] >= need;
                    if (fits ? (!best_fits || cap[s] < cap[best]) : (!best_fits && cap[s] > cap[best])) best = s;
                }
            }
            if (best < 0) best = n_slots++, cap[best] = 0;
            if (cap[best] < need) cap[best] = need;
            slot_free[best] = 0;
            slot_of[r] = best;
        }
        /* release everything whose last reader was op i */
        for (int t = 0; t < MAX_TENSORS; t++) {
            if (slot_of[t] >= 0 && last[t] == i) slot_free[slot_of[t]] = 1;
        }
    }

    float* mem[MAX_TENSORS];
    size_t total = 0;
    for (int s = 0; s < n_slots; s++) mem[s] = v5_alloc(cap[s]), total += cap[s] * sizeof(float);
    for (int t = 0; t < MAX_TENSORS; t++) {
        int r = root_of(t);
        if (slot_of[r] >= 0) g_t[t].d = mem[slot_of[r]] + (t == r ? 0 : g_alias_off[t]);
    }
    return total;
}

static void load_plan(const char* path, const float* weights, int use_v5, int T, int* in_id) {
    FILE* f = fopen(path, "r");
    if (!f) die("cannot open plan");
    char word[32];
    while (fscanf(f, "%31s", word) == 1) {
        Op* o = &g_ops[g_nops];
        memset(o, 0, sizeof(*o));
        if (strcmp(word, "TENSOR") == 0) {
            int id, c, h, w;
            if (fscanf(f, "%d %d %d %d", &id, &c, &h, &w) != 4 || id >= MAX_TENSORS) die("bad TENSOR");
            g_t[id] = (Tensor){c, h, w, NULL};
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
            g_alias[a] = g_alias[b] = src + 1;
            g_alias_off[a] = 0;
            g_alias_off[b] = (size_t)g_t[a].c * g_t[a].h * g_t[a].w;
            continue;
        }
        if (strcmp(word, "CONV") == 0) {
            o->type = OP_CONV;
            if (fscanf(f, "%d %d %d %d %d %d %d %d %ld %ld", &o->out, &o->in, &o->cin, &o->cout,
                       &o->k, &o->s, &o->p, &o->m, &o->woff, &o->boff) != 10) die("bad CONV");
            o->use_v5 = use_v5 && o->k == 3 && o->s == 1;
            if (o->use_v5 && g_v6 && !g_plan_m) o->m = v6_pick_m(g_t[o->in].h + 2, g_t[o->in].w + 2, o->cin, o->cout, T);
            if (o->use_v5) {
                int alpha = o->m + 2;
                float* U = (float*)malloc((size_t)o->cout * o->cin * alpha * alpha * sizeof(float));
                transform_weights(o->m, o->cin, o->cout, weights + o->woff, U);
                o->pack = g_v6 ? v6_pack_weights(o->m, U, o->cin, o->cout) : v5_pack_weights(o->m, U, o->cin, o->cout);
                free(U);
            } else {
                o->pack = dp_pack_weights(weights + o->woff, o->cin, o->cout, o->k);
            }
            o->bias = o->boff >= 0 ? weights + o->boff : NULL;
            if (o->p) o->pad = pad_buffer(o->cin, g_t[o->in].h, g_t[o->in].w, o->p);
        } else if (strcmp(word, "DWCONV") == 0) {
            o->type = OP_CONV;
            o->depthwise = 1;
            if (fscanf(f, "%d %d %d %d %d %d %ld %ld", &o->out, &o->in, &o->cin, &o->k, &o->s, &o->p,
                       &o->woff, &o->boff) != 8) die("bad DWCONV");
            o->cout = o->cin;
            o->w = weights + o->woff;
            o->bias = o->boff >= 0 ? weights + o->boff : NULL;
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
            fc_pack_rows((float*)o->w, o->nin, o->nout);
            if (g_fc_fp16) {
#if HAS_ARM_NEON
                /* Narrow the grouped rows in place: each half lands before the float it came from */
                size_t n = (size_t)(o->nout / 8) * 8 * o->nin;
                __fp16* h = (__fp16*)o->w;
                for (size_t i = 0; i < n; i++) h[i] = (__fp16)o->w[i];
#else
                die("--fc-fp16 needs NEON");
#endif
            }
        } else {
            die("unknown plan op");
        }
        Op* prev = g_nops ? &g_ops[g_nops - 1] : NULL;
        if (g_fuse_act && prev && prev->type == OP_CONV && !prev->act && o->in == o->out && o->in == prev->out &&
            (o->type == OP_RELU || o->type == OP_CLIP || o->type == OP_SILU)) {
            prev->act = o->type == OP_SILU ? 2 : 1;
            prev->lo = o->type == OP_RELU ? 0.0f : o->lo;
            prev->hi = o->type == OP_RELU ? INFINITY : o->hi;
            continue;
        }
        if (++g_nops >= MAX_OPS) die("too many ops");
    }
    fclose(f);

    /* A v6 conv whose output only feeds the next op, a padded conv, writes into that conv's
     * padded input, which skips a copy of the whole tensor through memory. Nothing runs in
     * between, so the pad buffer (shared by convs of one shape) can't be overwritten, except
     * by the producer itself when it reads its own input from that buffer: then the
     * consumer gets a buffer of its own. */
    for (int i = 0; g_v6 && i + 1 < g_nops; i++) {
        Op *o = &g_ops[i], *c = &g_ops[i + 1];
        if (o->type != OP_CONV || !o->use_v5 || o->act == 2 || c->type != OP_CONV || c->depthwise || !c->p ||
            c->in != o->out || g_alias[o->out])
            continue;
        int readers = 0;
        for (int j = 0; j < g_nops; j++) {
            Op* q = &g_ops[j];
            readers += q->in == o->out || (q->type == OP_ADD && q->in2 == o->out);
            for (int k = 0; q->type == OP_CONCAT && k < q->n_ins; k++) readers += q->ins[k] == o->out && k > 0;
        }
        for (int t = 0; t < MAX_TENSORS; t++) readers += g_alias[t] == o->out + 1;
        for (int k = 0; k < g_n_outs; k++) readers += g_outs[k] == o->out;
        if (readers != 1) continue;
        if (c->pad == o->pad) {
            size_t len = (size_t)c->cin * (g_t[c->in].h + 2 * c->p) * (g_t[c->in].w + 2 * c->p);
            c->pad = v5_alloc(len);
            memset(c->pad, 0, len * sizeof(float));
        }
        o->out_pad = c->pad;
        o->out_p = c->p;
        c->pad_filled = 1;
    }
}

/* A 1x1 expansion read only by the 3x3 depthwise right after it runs inside that op
 * (conv2d_pw_dw), so its wide output is never stored: on large maps that tensor is
 * memory traffic, not compute. */
static void fuse_pw_dw(void) {
    int readers[MAX_TENSORS] = {0}, n = 0;
    for (int i = 0; i < g_nops; i++) {
        Op* o = &g_ops[i];
        if (o->type == OP_CONCAT) for (int j = 0; j < o->n_ins; j++) readers[o->ins[j]]++;
        else readers[o->in]++;
        if (o->type == OP_ADD) readers[o->in2]++;
    }
    for (int j = 0; j < g_n_outs; j++) readers[g_outs[j]]++;
    for (int i = 0; i < g_nops; i++) {
        Op* a = &g_ops[i], *b = &g_ops[i + 1];
        if (g_fuse_pwdw && i + 1 < g_nops && a->type == OP_CONV && !a->depthwise && !a->use_v5 && a->k == 1 &&
            a->s == 1 && a->p == 0 && a->act != 2 && b->type == OP_CONV && b->depthwise && b->k == 3 && b->p == 1 &&
            b->s <= 2 && b->act != 2 && b->in == a->out && readers[a->out] == 1 && !g_alias[a->out] &&
            g_t[a->in].h * g_t[a->in].w >= g_fuse_pwdw) {
            b->pw = a->pack;
            b->pw_bias = a->bias;
            b->pw_lo = a->act == 1 ? a->lo : -INFINITY;
            b->pw_hi = a->act == 1 ? a->hi : INFINITY;
            b->pw_cin = a->cin;
            b->in = a->in;
            continue;   /* a is dropped */
        }
        g_ops[n++] = *a;
    }
    g_nops = n;
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
        else if (strcmp(argv[i], "--kc") == 0) g_v5_kc = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--fc-fp16") == 0) g_fc_fp16 = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--recycle") == 0) g_recycle = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--fuse-act") == 0) g_fuse_act = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--dp-block") == 0) g_dp_block = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--dp-kci") == 0) g_dp_kci = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--k3") == 0) g_k3 = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--pw-block") == 0) g_pw_block = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--pw-kc") == 0) g_pw_kc = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--dw-c4-max") == 0) g_dw_c4_max = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--fuse-pwdw") == 0) g_fuse_pwdw = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--fuse-px") == 0) g_fuse_px = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--plan-m") == 0) g_plan_m = atoi(argv[i + 1]);
        else die("unknown argument");
    }
    if (!plan || !weights_path || !input_path || !output_path || !csv) die("missing argument");
    g_v6 = strcmp(variant, "v6_freq") == 0;
    if (!g_v6 && strcmp(variant, "v5_batched") != 0 && strcmp(variant, "direct_packed") != 0) die("unknown variant");

    float* weights = read_floats(weights_path, NULL);
    int in_id = -1;
    load_plan(plan, weights, strcmp(variant, "direct_packed") != 0, T, &in_id);
    if (in_id < 0 || g_n_outs == 0) die("plan has no INPUT or OUTPUT");
    fuse_pw_dw();
    size_t act_bytes = assign_buffers(in_id);
    fprintf(stderr, "activation memory: %.1f MB (recycle=%d)\n", act_bytes / 1e6, g_recycle);

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
