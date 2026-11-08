/*
 * Whole-model float32 engine for the GAP8 (and a laptop build that checks it).
 *
 * Activations live in HyperRAM (8 MB) and weights in HyperFlash (readfs). Each layer is
 * streamed through one L2 arena: a block of weights, then bands of input rows, are read
 * in, the cluster computes the band, and the output rows are written back. 3x3 stride-1
 * convs use Winograd F(4,3) (weights transformed on the chip, a block at a time, because
 * transformed weights would not fit the flash); everything else is direct.
 *
 * The platform file defines, before including this:
 *   ram_rd(off, dst, bytes), ram_wr(off, src, bytes)   HyperRAM <-> L2
 *   fl_rd(off, dst, bytes)                              weights file -> L2 (bytes > 0)
 *   fl_rd_u(off, dst, bytes)                            pretransformed Winograd weights -> L2
 *   par(fn)                                             run fn(core, NC) on every core
 *   now_us()                                            wall clock, microseconds
 * and includes the generated net header (TENS, OPS, N_OPS, IN_ID, OUTS, N_OUTS).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

/* op types and Ten/Op come from the net header */
enum { CAT_WINO, CAT_CONV, CAT_DW, CAT_POOL, CAT_ELT, CAT_GEMM, N_CAT };
static const char *CAT_NAME[N_CAT] = {"conv3x3_s1", "conv_other", "depthwise", "pool", "eltwise", "gemm"};

static int use_wino = 1;   /* 0: every conv direct, for comparison */
#define CC 8   /* input channels per Winograd transform step */

static float *A;      /* the L2 arena */
static int A_n;       /* its size in floats */
static int NC = 8;
static uint32_t io_us, cat_us[N_CAT], op_us[N_OPS], wt_us;
static float *L1S;   /* per-core Winograd sums and transformed inputs, in cluster L1 */
static int L1S_n;   /* its size in floats */
static uint64_t cat_macs[N_CAT];
static int fail;   /* a layer did not fit L2 */
#define FAIL(msg) do { printf("FAIL: %s (op %d)\n", msg, (int)(o - OPS)); fail = 1; return; } while (0)

static void timed_ram_rd(uint32_t off, float *dst, int n) { uint32_t t = now_us(); ram_rd(off, dst, n * 4); io_us += now_us() - t; }
static void timed_ram_wr(uint32_t off, const float *src, int n) { uint32_t t = now_us(); ram_wr(off, src, n * 4); io_us += now_us() - t; }
static void timed_fl_rd(int off, float *dst, int n) { if (n <= 0) return; uint32_t t = now_us(); fl_rd((uint32_t)off * 4, dst, n * 4); io_us += now_us() - t; }
static void timed_fl_rd_u(int off, float *dst, int n) { uint32_t t = now_us(); fl_rd_u((uint32_t)off * 4, dst, n * 4); io_us += now_us() - t; }

/* exp from the Cephes polynomial, so the laptop and the chip get the same bits */
static float ex(float x) {
    if (x > 88.3762626647949f) x = 88.3762626647949f;
    if (x < -88.3762626647949f) x = -88.3762626647949f;
    float fx = x * 1.44269504088896341f;
    int e = (int)(fx >= 0 ? fx + 0.5f : fx - 0.5f);
    float ef = (float)e;
    float r = x - ef * 0.693359375f;
    r = r - ef * -2.12194440e-4f;
    float p = 1.9875691500E-4f;
    p = p * r + 1.3981999507E-3f;
    p = p * r + 8.3334519073E-3f;
    p = p * r + 4.1665795894E-2f;
    p = p * r + 1.6666665459E-1f;
    p = p * r + 5.0000001201E-1f;
    p = p * r * r + r + 1.0f;
    union { int32_t i; float f; } s = {(e + 127) << 23};
    return p * s.f;
}

static inline float act(float v, int a, float lo, float hi) {
    if (a == ACT_CLAMP) return v < lo ? lo : (v > hi ? hi : v);
    if (a == ACT_SILU) return v / (1.0f + ex(-v));
    return v;
}

/* Channels [c0, c0 + cc) and rows [y0, y0 + rows) of tensor t into dst[cc][rows][wp], with input
 * column 0 at column `left`. Anything outside the tensor is `fill`. */
static void load_band(const Ten *t, int c0, int cc, int y0, int rows, int wp, int left, float fill, float *dst) {
    int W = t->w;
    if (left == 0 && wp == W && y0 >= 0 && y0 + rows <= t->h) {
        for (int c = 0; c < cc; c++)
            timed_ram_rd(t->off + ((uint32_t)(c0 + c) * t->h + y0) * W * 4, dst + (size_t)c * rows * W, rows * W);
        return;
    }
    for (int c = 0; c < cc; c++) for (int r = 0; r < rows; r++) {
        float *d = dst + ((size_t)c * rows + r) * wp;
        int y = y0 + r;
        if (y < 0 || y >= t->h) { for (int i = 0; i < wp; i++) d[i] = fill; continue; }
        for (int i = 0; i < left; i++) d[i] = fill;
        for (int i = left + W; i < wp; i++) d[i] = fill;
        timed_ram_rd(t->off + ((uint32_t)(c0 + c) * t->h + y) * W * 4, d + left, W);
    }
}

/* Out[cc][rows][Wo] -> rows [y0, y0 + rows) of channels [c0, c0 + cc) */
static void store_band(const Ten *t, int c0, int cc, int y0, int rows, const float *src) {
    for (int c = 0; c < cc; c++)
        timed_ram_wr(t->off + ((uint32_t)(c0 + c) * t->h + y0) * t->w * 4, src + (size_t)c * rows * t->w, rows * t->w);
}

/* ---------------- kernel arguments (set by the FC, read by the cores) ---------------- */
static struct {
    const Op *o;
    const float *in, *w, *b, *in2;
    float *out, *u, *scratch;
    int n, cin, rows, rin, wp, Wo, k, s, nco;
    float fill;
} J;

/* ---------------- Winograd F(4,3) ---------------- */
static const float G4[6][3] = {{1 / 4.f, 0, 0}, {-1 / 6.f, -1 / 6.f, -1 / 6.f}, {-1 / 6.f, 1 / 6.f, -1 / 6.f},
                               {1 / 24.f, 1 / 12.f, 1 / 6.f}, {1 / 24.f, -1 / 12.f, 1 / 6.f}, {0, 0, 1}};

static inline void bt4(const float *d, int s, float *o, int os) {
    float d0 = d[0], d1 = d[s], d2 = d[2 * s], d3 = d[3 * s], d4 = d[4 * s], d5 = d[5 * s];
    float p = d3 + d4, q = d1 + d2, r = d4 - d3, u = d1 - d2, w = d4 - d2, x = d3 - d1;
    o[0] = d0 * 4.0f - d2 * 5.0f + d4;  o[os] = p - q * 4.0f;     o[2 * os] = r + u * 4.0f;
    o[3 * os] = w + x * 2.0f;           o[4 * os] = w - x * 2.0f; o[5 * os] = d1 * 4.0f - d3 * 5.0f + d5;
}
static inline void at4(const float *m, int s, float *o, int os) {
    float m1 = m[s], m2 = m[2 * s], m3 = m[3 * s], m4 = m[4 * s];
    float a = m1 + m2, b = m1 - m2, c = m3 + m4, d = m3 - m4;
    o[0] = m[0] + a + c; o[os] = b + d * 2.0f; o[2 * os] = a + c * 4.0f; o[3 * os] = b + d * 8.0f + m[5 * s];
}

/* U[c][ci][36] = G g G^T for the block's n output channels */
static void k_wtrans(int core, int nc) {
    for (int i = core; i < J.n * J.cin; i += nc) {
        const float *g = J.w + (size_t)i * 9; float t[18], *o = J.u + (size_t)i * 36;
        for (int r = 0; r < 6; r++) for (int c = 0; c < 3; c++)
            t[r * 3 + c] = G4[r][0] * g[c] + G4[r][1] * g[3 + c] + G4[r][2] * g[6 + c];
        for (int r = 0; r < 6; r++) for (int c = 0; c < 6; c++)
            o[r * 6 + c] = t[r * 3] * G4[c][0] + t[r * 3 + 1] * G4[c][1] + t[r * 3 + 2] * G4[c][2];
    }
}

/* Each core takes whole 4x4 output tiles of the band: CC input channels are transformed at a
 * time and multiplied into every output channel's 36 sums. With fewer tiles than cores, a tile's
 * output channels are split into groups so every core has work (each group redoes the tile's
 * input transforms). */
static void k_wino(int core, int nc) {
    const Op *o = J.o;
    int Tw = (J.Wo + 3) / 4, Th = (J.rows + 3) / 4, wp = J.wp, rin = J.rin, Cin = J.cin;
    float *M = J.scratch + (size_t)core * (J.nco * 36 + CC * 36), *V = M + J.nco * 36;
    int tiles = Th * Tw, G = (nc + tiles - 1) / tiles;
    if (G > J.n) G = J.n;
    for (int it = core; it < tiles * G; it += nc) {
        int t = it % tiles, g = it / tiles, ty = t / Tw, tx = t % Tw;
        int c0 = J.n * g / G, c1 = J.n * (g + 1) / G;
        for (int i = c0 * 36; i < c1 * 36; i++) M[i] = 0.0f;
        for (int ci0 = 0; ci0 < Cin; ci0 += CC) {
            int cc = Cin - ci0 < CC ? Cin - ci0 : CC;
            for (int j = 0; j < cc; j++) {
                const float *p = J.in + ((size_t)(ci0 + j) * rin + ty * 4) * wp + tx * 4;
                float s[36];
                for (int c = 0; c < 6; c++) bt4(p + c, wp, s + c, 6);
                for (int r = 0; r < 6; r++) bt4(s + r * 6, 1, V + j * 36 + r * 6, 1);
            }
            for (int c = c0; c < c1; c++) {
                const float *u = J.u + ((size_t)c * Cin + ci0) * 36; float *m = M + c * 36;
                for (int j = 0; j < cc; j++) for (int q = 0; q < 36; q++) m[q] += u[j * 36 + q] * V[j * 36 + q];
            }
        }
        for (int c = c0; c < c1; c++) {
            float s[24], y[16], b = J.b ? J.b[c] : 0.0f;
            for (int q = 0; q < 6; q++) at4(M + c * 36 + q, 6, s + q, 6);
            for (int r = 0; r < 4; r++) at4(s + r * 6, 1, y + r * 4, 1);
            for (int r = 0; r < 4; r++) {
                int oy = ty * 4 + r;
                if (oy >= J.rows) break;
                float *dst = J.out + ((size_t)c * J.rows + oy) * J.Wo;
                for (int q = 0; q < 4; q++) {
                    int ox = tx * 4 + q;
                    if (ox < J.Wo) dst[ox] = act(y[r * 4 + q] + b, o->act, o->lo, o->hi);
                }
            }
        }
    }
}

/* Direct conv: one (output channel, output row) per work item */
static void k_direct(int core, int nc) {
    const Op *o = J.o;
    int k = J.k, s = J.s, wp = J.wp, rin = J.rin, kk = k * k;
    for (int it = core; it < J.n * J.rows; it += nc) {
        int c = it / J.rows, r = it % J.rows;
        const float *w = J.w + (size_t)c * J.cin * kk;
        float b = J.b ? J.b[c] : 0.0f, *y = J.out + ((size_t)c * J.rows + r) * J.Wo;
        for (int x = 0; x < J.Wo; x++) {
            float acc = 0.0f;
            for (int ci = 0; ci < J.cin; ci++) {
                const float *p = J.in + ((size_t)ci * rin + r * s) * wp + x * s, *wk = w + ci * kk;
                for (int kh = 0; kh < k; kh++) for (int kw = 0; kw < k; kw++) acc += p[kh * wp + kw] * wk[kh * k + kw];
            }
            y[x] = act(acc + b, o->act, o->lo, o->hi);
        }
    }
}

/* Depthwise conv (J.w set) or max pool (J.w NULL) over a band of J.n channels */
static void k_window(int core, int nc) {
    const Op *o = J.o;
    int k = J.k, s = J.s, wp = J.wp, rin = J.rin;
    for (int it = core; it < J.n * J.rows; it += nc) {
        int c = it / J.rows, r = it % J.rows;
        const float *w = J.w ? J.w + (size_t)c * k * k : NULL;
        float b = J.b ? J.b[c] : 0.0f, *y = J.out + ((size_t)c * J.rows + r) * J.Wo;
        for (int x = 0; x < J.Wo; x++) {
            const float *p = J.in + ((size_t)c * rin + r * s) * wp + x * s;
            float v;
            if (w) {
                v = 0.0f;
                for (int kh = 0; kh < k; kh++) for (int kw = 0; kw < k; kw++) v += p[kh * wp + kw] * w[kh * k + kw];
                v = act(v + b, o->act, o->lo, o->hi);
            } else {
                v = -INFINITY;
                for (int kh = 0; kh < k; kh++) for (int kw = 0; kw < k; kw++) if (p[kh * wp + kw] > v) v = p[kh * wp + kw];
            }
            y[x] = v;
        }
    }
}

static void k_elt(int core, int nc) {
    const Op *o = J.o;
    for (int i = core; i < J.n; i += nc) {
        float v = J.in[i];
        if (o->type == OP_ADD) v += J.in2[i];
        else if (o->type == OP_RELU) v = v < 0.0f ? 0.0f : v;
        else if (o->type == OP_CLIP) v = act(v, ACT_CLAMP, o->lo, o->hi);
        else v = act(v, ACT_SILU, 0, 0);
        J.out[i] = v;
    }
}

static void k_gap(int core, int nc) {
    for (int c = core; c < J.n; c += nc) {
        const float *p = J.in + (size_t)c * J.rows;
        float s = 0.0f;
        for (int i = 0; i < J.rows; i++) s += p[i];
        J.out[c] = s / (float)J.rows;
    }
}

static void k_gemm(int core, int nc) {
    for (int r = core; r < J.n; r += nc) {
        const float *w = J.w + (size_t)r * J.cin;
        float s = 0.0f;
        for (int i = 0; i < J.cin; i++) s += w[i] * J.in[i];
        J.out[r] = s + (J.b ? J.b[r] : 0.0f);
    }
}

/* ---------------- layer schedules ---------------- */
/* Winograd's cost per (output, input channel) pair: weight transform (~324 flops, redone for every
 * run because transformed weights don't fit the flash), 72 per tile multiplied, and the input
 * transforms (~150 per tile) shared by a block of nco outputs. Direct is 18 per output pixel.
 * ponytail: flop counts only; time both on the board if a layer sits near the line. */
static int wino_pays(const Op *o) {
    if (!use_wino || o->type != OP_CONV || o->k != 3 || o->s != 1 || o->p != 1) return 0;
    const Ten *to = &TENS[o->out];
    int Cin = o->cin, Wo = to->w, Ho = to->h, wp = (Wo + 3) / 4 * 4 + 2;
    int pre = o->uoff >= 0;
    int per_co = (pre ? 0 : Cin * 9) + Cin * 36 + 1 + 4 * 4 * Wo;   /* weights, U, bias, one band of output */
    int nco = L1S ? (A_n - Cin * 6 * wp) / per_co : (A_n - Cin * 6 * wp - NC * CC * 36) / (per_co + NC * 36);
    if (nco > o->cout) nco = o->cout;
    if (L1S && nco > L1S_n / (NC * 36) - CC) nco = L1S_n / (NC * 36) - CC;
    if (nco < 1) return 0;
    /* Measured on ResNet-18 (board): with the transform done on the chip, 14x14 layers still gain
     * 1.35x but 7x7 ones lose (0.54x): too few tiles, and the transform and weight reads dominate. */
    if (!pre && Ho * Wo < 100) return 0;
    double tiles = (double)((Ho + 3) / 4) * ((Wo + 3) / 4);
    return (pre ? 0.0 : 324.0) + tiles * 72.0 + tiles * 150.0 / nco < 18.0 * Ho * Wo;
}

static void run_conv(const Op *o) {
    const Ten *ti = &TENS[o->in], *to = &TENS[o->out];
    int k = o->k, s = o->s, p = o->p, Cin = o->cin, Cout = o->cout, Ho = to->h, Wo = to->w;
    int wino = wino_pays(o);
    int kk = k * k, wp = wino ? (Wo + 3) / 4 * 4 + 2 : ti->w + 2 * p;
    int pre = wino && o->uoff >= 0;   /* transformed weights come from the flash */
    int wsz = pre ? 0 : Cin * kk, usz = wino ? Cin * 36 : 0, step = wino ? 4 : 1;
    /* floats needed for a block of nco output channels and a band of nr output rows */
#define RIN(nr) (wino ? (nr) + 2 : ((nr) - 1) * s + k)
#define NEED(nco, nr) ((nco) * (wsz + usz + 1) + Cin * RIN(nr) * wp + (nco) * (nr) * Wo + (wino && !L1S ? NC * ((nco) + CC) * 36 : 0))
    int nco = Cout;
    if (wino && L1S && nco > L1S_n / (NC * 36) - CC) nco = L1S_n / (NC * 36) - CC;
    while (nco > 1 && (NEED(nco, step) > A_n || nco * (wsz + usz) > A_n / 2)) nco = (nco + 1) / 2;
    if (NEED(nco, step) > A_n) { FAIL("conv does not fit L2"); }
    int nr = (Ho + step - 1) / step * step;
    while (nr > step && NEED(nco, nr) > A_n) nr -= step;
    float *Wb = A, *Ub = Wb + nco * wsz, *Bb = Ub + nco * usz, *In = Bb + nco, *Out = In + Cin * RIN(nr) * wp;
    float *Sc = wino && L1S ? L1S : Out + nco * nr * Wo;
    J.o = o; J.cin = Cin; J.wp = wp; J.Wo = Wo; J.k = k; J.s = s; J.nco = nco; J.scratch = Sc;
    for (int co0 = 0; co0 < Cout; co0 += nco) {
        int n = Cout - co0 < nco ? Cout - co0 : nco;
        if (pre) timed_fl_rd_u(o->uoff + co0 * usz, Ub, n * usz);
        else timed_fl_rd(o->woff + co0 * wsz, Wb, n * wsz);
        if (o->boff >= 0) timed_fl_rd(o->boff + co0, Bb, n);
        J.n = n; J.w = Wb; J.u = Ub; J.b = o->boff >= 0 ? Bb : NULL;
        if (wino && !pre) { uint32_t t = now_us(); par(k_wtrans); wt_us += now_us() - t; }
        for (int y0 = 0; y0 < Ho; y0 += nr) {
            int rows = Ho - y0 < nr ? Ho - y0 : nr, band = wino ? (rows + 3) / 4 * 4 : rows;
            J.rows = rows; J.rin = RIN(band); J.in = In; J.out = Out;
            load_band(ti, 0, Cin, y0 * s - p, J.rin, wp, p, 0.0f, In);
            par(wino ? k_wino : k_direct);
            store_band(to, co0, n, y0, rows, Out);
        }
    }
    cat_macs[wino ? CAT_WINO : CAT_CONV] += (uint64_t)Cout * Cin * kk * Ho * Wo;
#undef RIN
#undef NEED
}

/* depthwise conv and max pool: blocks of channels, bands of rows */
static void run_window(const Op *o) {
    const Ten *ti = &TENS[o->in], *to = &TENS[o->out];
    int dw = o->type == OP_DWCONV, C = to->c, k = o->k, s = o->s, p = o->p, Ho = to->h, Wo = to->w;
    int wp = ti->w + 2 * p, wfl = dw ? C * k * k + (o->boff >= 0 ? C : 0) : 0;
#define RIN(nr) (((nr) - 1) * s + k)
    int avail = A_n - wfl, cc = C, nr = Ho;
    while (cc > 1 && cc * (RIN(1) * wp + Wo) > avail) cc = (cc + 1) / 2;
    while (nr > 1 && cc * (RIN(nr) * wp + nr * Wo) > avail) nr = (nr + 1) / 2;
    if (cc * (RIN(nr) * wp + nr * Wo) > avail) { FAIL("window op does not fit L2"); }
    float *Wb = A, *In = A + wfl, *Out = In + cc * RIN(nr) * wp;
    if (dw) {
        timed_fl_rd(o->woff, Wb, C * k * k);
        if (o->boff >= 0) timed_fl_rd(o->boff, Wb + C * k * k, C);
    }
    J.o = o; J.k = k; J.s = s; J.wp = wp; J.Wo = Wo;
    for (int c0 = 0; c0 < C; c0 += cc) {
        int n = C - c0 < cc ? C - c0 : cc;
        J.n = n;
        J.w = dw ? Wb + (size_t)c0 * k * k : NULL;
        J.b = dw && o->boff >= 0 ? Wb + C * k * k + c0 : NULL;
        for (int y0 = 0; y0 < Ho; y0 += nr) {
            int rows = Ho - y0 < nr ? Ho - y0 : nr;
            J.rows = rows; J.rin = RIN(rows); J.in = In; J.out = Out;
            load_band(ti, c0, n, y0 * s - p, J.rin, wp, p, dw ? 0.0f : -INFINITY, In);
            par(k_window);
            store_band(to, c0, n, y0, rows, Out);
        }
    }
    if (dw) cat_macs[CAT_DW] += (uint64_t)C * k * k * Ho * Wo;
#undef RIN
}

static int numel(const Ten *t) { return t->c * t->h * t->w; }

/* ADD, RELU, CLIP, SILU: flat chunks */
static void run_elt(const Op *o) {
    int N = numel(&TENS[o->out]), two = o->type == OP_ADD, chunk = A_n / (two ? 2 : 1);
    J.o = o;
    for (int i0 = 0; i0 < N; i0 += chunk) {
        int n = N - i0 < chunk ? N - i0 : chunk;
        timed_ram_rd(TENS[o->in].off + i0 * 4, A, n);
        if (two) timed_ram_rd(TENS[o->in2].off + i0 * 4, A + chunk, n);
        J.n = n; J.in = A; J.in2 = A + chunk; J.out = A;
        par(k_elt);
        timed_ram_wr(TENS[o->out].off + i0 * 4, A, n);
    }
}

static void ram_copy(uint32_t dst, uint32_t src, int N) {
    for (int i0 = 0; i0 < N; i0 += A_n) {
        int n = N - i0 < A_n ? N - i0 : A_n;
        timed_ram_rd(src + i0 * 4, A, n);
        timed_ram_wr(dst + i0 * 4, A, n);
    }
}

static void run_concat(const Op *o) {
    uint32_t dst = TENS[o->out].off;
    for (int i = 0; i < o->n_ins; i++) {
        const Ten *t = &TENS[o->ins[i]];
        ram_copy(dst, t->off, numel(t));
        dst += numel(t) * 4;
    }
}

static void run_upsample(const Op *o) {
    const Ten *ti = &TENS[o->in], *to = &TENS[o->out];
    int W = ti->w, rows = A_n / (5 * W);   /* rows in, plus 4 output values per input value */
    if (rows < 1) { FAIL("upsample row does not fit L2"); }
    if (rows > ti->h) rows = ti->h;
    for (int c = 0; c < ti->c; c++) for (int y0 = 0; y0 < ti->h; y0 += rows) {
        int n = ti->h - y0 < rows ? ti->h - y0 : rows;
        float *in = A, *out = A + n * W;
        timed_ram_rd(ti->off + ((uint32_t)c * ti->h + y0) * W * 4, in, n * W);
        for (int r = 0; r < n; r++) for (int x = 0; x < W; x++) {
            float v = in[r * W + x], *d = out + (size_t)r * 2 * (2 * W);
            d[2 * x] = d[2 * x + 1] = d[2 * W + 2 * x] = d[2 * W + 2 * x + 1] = v;
        }
        timed_ram_wr(to->off + ((uint32_t)c * to->h + y0 * 2) * to->w * 4, out, n * 4 * W);
    }
}

static void run_gap(const Op *o) {
    const Ten *ti = &TENS[o->in];
    int plane = ti->h * ti->w, cc = (A_n - ti->c) / plane;
    if (cc < 1) { FAIL("gap plane does not fit L2"); }
    float *res = A + (size_t)cc * plane;
    J.rows = plane; J.in = A;
    for (int c0 = 0; c0 < ti->c; c0 += cc) {
        int n = ti->c - c0 < cc ? ti->c - c0 : cc;
        timed_ram_rd(ti->off + (uint32_t)c0 * plane * 4, A, n * plane);
        J.n = n; J.out = res + c0;
        par(k_gap);
    }
    timed_ram_wr(TENS[o->out].off, res, ti->c);
}

static void run_gemm(const Op *o) {
    int nin = o->nin, nout = o->nout;
    float *x = A, *y = A + nin, *b = y + nout, *w = b + nout;
    int rows = (A_n - nin - 2 * nout) / nin;
    if (rows < 1) { FAIL("gemm row does not fit L2"); }
    timed_ram_rd(TENS[o->in].off, x, nin);
    if (o->boff >= 0) timed_fl_rd(o->boff, b, nout);
    J.in = x; J.cin = nin;
    for (int r0 = 0; r0 < nout; r0 += rows) {
        int n = nout - r0 < rows ? nout - r0 : rows;
        timed_fl_rd(o->woff + r0 * nin, w, n * nin);
        J.n = n; J.w = w; J.out = y + r0; J.b = o->boff >= 0 ? b + r0 : NULL;
        par(k_gemm);
    }
    timed_ram_wr(TENS[o->out].off, y, nout);
    cat_macs[CAT_GEMM] += (uint64_t)nin * nout;
}

static int op_cat(const Op *o) {
    switch (o->type) {
    case OP_CONV: return wino_pays(o) ? CAT_WINO : CAT_CONV;
    case OP_DWCONV: return CAT_DW;
    case OP_MAXPOOL: case OP_GAP: return CAT_POOL;
    case OP_GEMM: return CAT_GEMM;
    default: return CAT_ELT;
    }
}

static void run_net(void) {
    memset(cat_us, 0, sizeof cat_us);
    memset(cat_macs, 0, sizeof cat_macs);
    io_us = wt_us = 0;
    for (int i = 0; i < N_OPS; i++) {
        const Op *o = &OPS[i];
        uint32_t t0 = now_us();
        switch (o->type) {
        case OP_CONV: run_conv(o); break;
        case OP_DWCONV: case OP_MAXPOOL: run_window(o); break;
        case OP_GAP: run_gap(o); break;
        case OP_GEMM: run_gemm(o); break;
        case OP_CONCAT: run_concat(o); break;
        case OP_UPSAMPLE: run_upsample(o); break;
        default: run_elt(o); break;
        }
        op_us[i] = now_us() - t0;
#ifdef OPHASH
        {   /* debug: hash of every op's output */
            const Ten *t = &TENS[o->out]; uint32_t h = 2166136261u;
            for (int i0 = 0; i0 < numel(t); i0 += A_n) {
                int n = numel(t) - i0 < A_n ? numel(t) - i0 : A_n;
                ram_rd(t->off + i0 * 4, A, n * 4);
                for (int j = 0; j < n * 4; j++) h = (h ^ ((uint8_t *)A)[j]) * 16777619u;
            }
            printf("op %d %08x\n", i, (unsigned)h);
        }
#endif
        cat_us[op_cat(o)] += op_us[i];
        if (fail) return;
    }
}

/* FNV-1a over the output's bytes, the largest error against the reference file, and its largest value */
static void check_output(const Ten *t, int ref_off, uint32_t *hash, float *err, float *mx) {
    int N = numel(t), chunk = A_n / 2;
    for (int i0 = 0; i0 < N; i0 += chunk) {
        int n = N - i0 < chunk ? N - i0 : chunk;
        float *y = A, *r = A + chunk;
        timed_ram_rd(t->off + i0 * 4, y, n);
        fl_rd_ref((uint32_t)(ref_off + i0) * 4, r, n * 4);
        for (int i = 0; i < n; i++) {
            const uint8_t *b = (const uint8_t *)&y[i];
            for (int j = 0; j < 4; j++) *hash = (*hash ^ b[j]) * 16777619u;
            float d = fabsf(y[i] - r[i]), a = fabsf(r[i]);
            if (d > *err || d != d) *err = d;
            if (a > *mx) *mx = a;
        }
    }
}
