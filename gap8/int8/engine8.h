/*
 * 8-bit integer engine for the GAP8, same integer arithmetic as DORY/PULP-NN (Quantlab models).
 *
 * Activations are HWC bytes in HyperRAM. For each layer the fabric controller stages a band of
 * input rows (zero padding built in) and a block of weights in L2; a cluster task copies them into
 * L1 with the cluster DMA, the 8 cores compute an output tile there, and the tile goes back to an L2
 * output band, which the fabric controller writes to HyperRAM.
 *
 * Dense convs: each core takes output pixels two at a time, builds their patches (im2col; 1x1 convs
 * read the input in place), and computes 4 output channels x 2 pixels with 4-way 8-bit dot products.
 *
 * Platform hooks (defined before including this): ram_rd/ram_wr (HyperRAM <-> L2, bytes),
 * fl_rd (weights file -> L2), act_l2 / w_l2 (L2 copies or NULL), cl_run(pre, fn, post) (pre and post on core 0 with barriers around fn
 * on every core), dma_in(l1, l2, n), dma_out2d(l2, l1, size, stride, len), now_us(), and SDOT4U/SDOT4S.
 */
#include <stdint.h>
#include <string.h>

enum { C8_CONV, C8_DW, C8_ADD, C8_GAP, C8_FC, N_C8 };
static const char *C8_NAME[N_C8] = {"conv", "depthwise", "add", "pool", "fc"};

static uint8_t *A2;          /* L2 staging arena */
static int A2_n;
static uint8_t *A1;          /* cluster L1 buffer */
static int A1_n;
static int NC8 = 8;
static uint32_t io8_us, cat8_us[N_C8], lay_us[N_LAYERS];
static int fail8;
static void (*after_layer8)(int);   /* debug hook: called after each layer */
#define FAIL8(m) do { printf("FAIL: %s (layer %d)\n", m, (int)(L - LAYERS)); fail8 = 1; return; } while (0)

#define AL4(x) (((x) + 3) & ~3)
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---------------- job (set by the FC / core 0, read by every core) ---------------- */
static struct {
    const Layer *L;
    /* L1 side */
    uint8_t *in1, *w1, *out1, *col1;
    const int32_t *kl;              /* k, l pairs for the block's channels */
    int rows, rin, wp, co0, nco, cout, col_per_core;
    int n;                          /* elementwise / fc counts */
} J8;

/* ---------------- dense conv: im2col, 4 channels x 2 pixels ---------------- */
static inline const uint8_t *patch(const Layer *L, int r, int x, uint8_t *buf) {
    int cin = L->cin, k = L->k, s = L->s;
    const uint8_t *base = J8.in1 + ((size_t)(r * s) * J8.wp + x * s) * cin;
    if (k == 1) return base;
    int row = k * cin;
    if ((row & 3) == 0) {          /* rows of whole words: copy words (aligned, since cin % 4 == 0) */
        uint32_t *d = (uint32_t *)buf;
        for (int kh = 0; kh < k; kh++) {
            const uint32_t *src = (const uint32_t *)(base + (size_t)kh * J8.wp * cin);
            for (int i = 0; i < row / 4; i++) *d++ = src[i];
        }
        return buf;
    }
    for (int kh = 0; kh < k; kh++) for (int i = 0; i < row; i++) buf[kh * row + i] = base[(size_t)kh * J8.wp * cin + i];
    for (int i = k * row; i < L->kpad; i++) buf[i] = 0;
    return buf;
}

static inline uint8_t rq8(int32_t acc, const int32_t *kl, int d, int lo, int hi) {
    return (uint8_t)clampi((acc * kl[0] + kl[1]) >> d, lo, hi);
}

static void k_conv(int core, int nc) {
    const Layer *L = J8.L;
    int Wo = L->w, np = J8.rows * Wo, kp = L->kpad, nco = J8.nco, sgn = L->in_signed;
    uint8_t *c0 = J8.col1 + (size_t)core * 2 * kp, *c1 = c0 + kp;
    for (int p = core * 2; p < np; p += nc * 2) {
        int two = p + 1 < np;
        const uint8_t *a0 = patch(L, p / Wo, p % Wo, c0);
        const uint8_t *a1 = two ? patch(L, (p + 1) / Wo, (p + 1) % Wo, c1) : a0;
        uint8_t *y0 = J8.out1 + (size_t)p * nco, *y1 = y0 + nco;
        int co = 0;
        for (; co + 4 <= nco; co += 4) {
            const uint8_t *w = J8.w1 + (size_t)co * kp;
            int32_t s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
            for (int i = 0; i < kp; i += 4) {
                uint32_t x0 = *(const uint32_t *)(a0 + i), x1 = *(const uint32_t *)(a1 + i);
                uint32_t w0 = *(const uint32_t *)(w + i), w1 = *(const uint32_t *)(w + kp + i);
                uint32_t w2 = *(const uint32_t *)(w + 2 * kp + i), w3 = *(const uint32_t *)(w + 3 * kp + i);
                if (sgn) {
                    s00 = SDOT4S(x0, w0, s00); s01 = SDOT4S(x0, w1, s01); s02 = SDOT4S(x0, w2, s02); s03 = SDOT4S(x0, w3, s03);
                    s10 = SDOT4S(x1, w0, s10); s11 = SDOT4S(x1, w1, s11); s12 = SDOT4S(x1, w2, s12); s13 = SDOT4S(x1, w3, s13);
                } else {
                    s00 = SDOT4U(x0, w0, s00); s01 = SDOT4U(x0, w1, s01); s02 = SDOT4U(x0, w2, s02); s03 = SDOT4U(x0, w3, s03);
                    s10 = SDOT4U(x1, w0, s10); s11 = SDOT4U(x1, w1, s11); s12 = SDOT4U(x1, w2, s12); s13 = SDOT4U(x1, w3, s13);
                }
            }
            const int32_t *kl = J8.kl + co * 2;
            y0[co] = rq8(s00, kl, L->d, L->lo, L->hi); y0[co + 1] = rq8(s01, kl + 2, L->d, L->lo, L->hi);
            y0[co + 2] = rq8(s02, kl + 4, L->d, L->lo, L->hi); y0[co + 3] = rq8(s03, kl + 6, L->d, L->lo, L->hi);
            if (two) {
                y1[co] = rq8(s10, kl, L->d, L->lo, L->hi); y1[co + 1] = rq8(s11, kl + 2, L->d, L->lo, L->hi);
                y1[co + 2] = rq8(s12, kl + 4, L->d, L->lo, L->hi); y1[co + 3] = rq8(s13, kl + 6, L->d, L->lo, L->hi);
            }
        }
        for (; co < nco; co++) {          /* channels past the last group of 4 */
            const uint8_t *w = J8.w1 + (size_t)co * kp;
            int32_t s0 = 0, s1 = 0;
            for (int i = 0; i < kp; i += 4) {
                uint32_t wv = *(const uint32_t *)(w + i);
                s0 = sgn ? SDOT4S(*(const uint32_t *)(a0 + i), wv, s0) : SDOT4U(*(const uint32_t *)(a0 + i), wv, s0);
                s1 = sgn ? SDOT4S(*(const uint32_t *)(a1 + i), wv, s1) : SDOT4U(*(const uint32_t *)(a1 + i), wv, s1);
            }
            y0[co] = rq8(s0, J8.kl + co * 2, L->d, L->lo, L->hi);
            if (two) y1[co] = rq8(s1, J8.kl + co * 2, L->d, L->lo, L->hi);
        }
    }
}

/* ---------------- Winograd F(2,3), 3x3 stride-1 dense convs ----------------
 * U' = (2G) g (2G)^T = 4U (int16, [co][16][ci], ci padded to even) comes from the host. Each core
 * takes two neighbouring 2x2 output tiles: their input tiles are transformed into V[2][16][ci] (int16,
 * B^T d B has no fractions), then 4 output channels x 2 tiles are accumulated over ci with 2-way
 * 16-bit dot products, one frequency at a time. A^T M A is 4x the direct sum exactly, so the result
 * after >> 2 is the same integer and requantizes the same. */
static int use_wino8 = 1;

static inline void bt2_tile(const uint8_t *p, int wp, int C, int cinp, int sgn, int16_t *V) {
    for (int ci = 0; ci < cinp; ci++) {
        int d[16], s[16];
        if (ci < C) {
            for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
                uint8_t v = p[((size_t)r * wp + c) * C + ci];
                d[r * 4 + c] = sgn ? (int8_t)v : v;
            }
        } else for (int i = 0; i < 16; i++) d[i] = 0;
        for (int c = 0; c < 4; c++) {          /* columns: B^T d */
            int d0 = d[c], d1 = d[4 + c], d2 = d[8 + c], d3 = d[12 + c];
            s[c] = d0 - d2; s[4 + c] = d1 + d2; s[8 + c] = d2 - d1; s[12 + c] = d1 - d3;
        }
        for (int r = 0; r < 4; r++) {          /* rows: (B^T d) B */
            int s0 = s[r * 4], s1 = s[r * 4 + 1], s2 = s[r * 4 + 2], s3 = s[r * 4 + 3];
            V[(r * 4 + 0) * cinp + ci] = s0 - s2; V[(r * 4 + 1) * cinp + ci] = s1 + s2;
            V[(r * 4 + 2) * cinp + ci] = s2 - s1; V[(r * 4 + 3) * cinp + ci] = s1 - s3;
        }
    }
}

static inline void at2_store(const int32_t *m, const Layer *L, const int32_t *kl, uint8_t *out, int oy, int ox, int rows, int Wo, int nco) {
    int32_t t[8];
    for (int c = 0; c < 4; c++) { t[c] = m[c] + m[4 + c] + m[8 + c]; t[4 + c] = m[4 + c] - m[8 + c] - m[12 + c]; }
    for (int r = 0; r < 2; r++) {
        int32_t y0 = t[r * 4] + t[r * 4 + 1] + t[r * 4 + 2], y1 = t[r * 4 + 1] - t[r * 4 + 2] - t[r * 4 + 3];
        if (oy + r >= rows) break;
        out[((size_t)(oy + r) * Wo + ox) * nco] = rq8(y0 >> 2, kl, L->d, L->lo, L->hi);
        if (ox + 1 < Wo) out[((size_t)(oy + r) * Wo + ox + 1) * nco] = rq8(y1 >> 2, kl, L->d, L->lo, L->hi);
    }
}

static void k_wino8(int core, int nc) {
    const Layer *L = J8.L;
    int C = L->cin, cinp = (C + 1) & ~1, Wo = L->w, nco = J8.nco, rows = J8.rows, wp = J8.wp, sgn = L->in_signed;
    int tw = (Wo + 1) / 2, th = (rows + 1) / 2, pairs = (tw + 1) / 2;
    int16_t *V0 = (int16_t *)(J8.col1 + (size_t)core * (2 * 16 * cinp * 2 + 512)), *V1 = V0 + 16 * cinp;
    int32_t *M = (int32_t *)(V1 + 16 * cinp);              /* [4 co][2 tiles][16] */
    const int16_t *U = (const int16_t *)J8.w1;
    for (int it = core; it < th * pairs; it += nc) {
        int ty = it / pairs, tx0 = (it % pairs) * 2, two = tx0 + 1 < tw;
        bt2_tile(J8.in1 + ((size_t)(2 * ty) * wp + 2 * tx0) * C, wp, C, cinp, sgn, V0);
        if (two) bt2_tile(J8.in1 + ((size_t)(2 * ty) * wp + 2 * tx0 + 2) * C, wp, C, cinp, sgn, V1);
        else memcpy(V1, V0, 16 * cinp * 2);
        for (int co = 0; co < nco; co += 4) {
            int nb = nco - co < 4 ? nco - co : 4;
            for (int f = 0; f < 16; f++) {
                const uint32_t *v0 = (const uint32_t *)(V0 + f * cinp), *v1 = (const uint32_t *)(V1 + f * cinp);
                const uint32_t *u0 = (const uint32_t *)(U + ((size_t)co * 16 + f) * cinp);
                const uint32_t *u1 = u0 + 16 * cinp / 2, *u2 = u1 + 16 * cinp / 2, *u3 = u2 + 16 * cinp / 2;
                int32_t a00 = 0, a01 = 0, a10 = 0, a11 = 0, a20 = 0, a21 = 0, a30 = 0, a31 = 0;
                if (nb == 4) {
                    for (int i = 0; i < cinp / 2; i++) {
                        uint32_t x0 = v0[i], x1 = v1[i];
                        a00 = SDOT2(u0[i], x0, a00); a01 = SDOT2(u0[i], x1, a01);
                        a10 = SDOT2(u1[i], x0, a10); a11 = SDOT2(u1[i], x1, a11);
                        a20 = SDOT2(u2[i], x0, a20); a21 = SDOT2(u2[i], x1, a21);
                        a30 = SDOT2(u3[i], x0, a30); a31 = SDOT2(u3[i], x1, a31);
                    }
                } else {
                    const uint32_t *uu[4] = {u0, u1, u2, u3};
                    int32_t *acc[4][2] = {{&a00, &a01}, {&a10, &a11}, {&a20, &a21}, {&a30, &a31}};
                    for (int j = 0; j < nb; j++) for (int i = 0; i < cinp / 2; i++) {
                        *acc[j][0] = SDOT2(uu[j][i], v0[i], *acc[j][0]);
                        *acc[j][1] = SDOT2(uu[j][i], v1[i], *acc[j][1]);
                    }
                }
                M[0 * 32 + f] = a00; M[0 * 32 + 16 + f] = a01; M[1 * 32 + f] = a10; M[1 * 32 + 16 + f] = a11;
                M[2 * 32 + f] = a20; M[2 * 32 + 16 + f] = a21; M[3 * 32 + f] = a30; M[3 * 32 + 16 + f] = a31;
            }
            for (int j = 0; j < nb; j++) {
                const int32_t *kl = J8.kl + (co + j) * 2;
                uint8_t *o = J8.out1 + co + j;
                at2_store(M + j * 32, L, kl, o, 2 * ty, 2 * tx0, rows, Wo, nco);
                if (two) at2_store(M + j * 32 + 16, L, kl, o, 2 * ty, 2 * tx0 + 2, rows, Wo, nco);
            }
        }
    }
}

/* depthwise: one (output row, channel range) per item, weights [kh][kw][c] */
static void k_dw(int core, int nc) {
    const Layer *L = J8.L;
    int C = L->c, k = L->k, s = L->s, Wo = L->w, sgn = L->in_signed;
    for (int it = core; it < J8.rows * Wo; it += nc) {
        int r = it / Wo, x = it % Wo;
        const uint8_t *base = J8.in1 + ((size_t)(r * s) * J8.wp + x * s) * C;
        uint8_t *y = J8.out1 + (size_t)it * C;
        for (int c = 0; c < C; c++) {
            int32_t acc = 0;
            for (int kh = 0; kh < k; kh++) for (int kw = 0; kw < k; kw++) {
                int v = sgn ? (int8_t)base[((size_t)kh * J8.wp + kw) * C + c] : base[((size_t)kh * J8.wp + kw) * C + c];
                acc += v * (int8_t)J8.w1[(kh * k + kw) * C + c];
            }
            y[c] = rq8(acc, J8.kl + c * 2, L->d, L->lo, L->hi);
        }
    }
}

/* residual add, each core a contiguous chunk; inputs that aren't rescaled use m = 1, a = 0, s = 0
 * with the clamp skipped (rq2 is never set in these models' exports, so one loop covers them) */
static void k_add(int core, int nc) {
    const Layer *L = J8.L;
    const int *p = L->p;
    int lo = L->lo, hi = L->hi, m1 = p[0], a1 = p[1], sh1 = p[2] & 255, q1 = p[2] >> 8;
    int m2 = p[3], a2 = p[4], sh2 = p[5] & 255, q2 = p[5] >> 8, om = p[6], oa = p[7], os = p[8];
    int s1 = L->in_signed & 1, s2 = L->in_signed >> 1;
    int per = (J8.n + nc - 1) / nc, i0 = core * per, i1 = i0 + per < J8.n ? i0 + per : J8.n;
    const uint8_t *a = J8.in1, *b = J8.in1 + J8.n;
    uint8_t *y = J8.out1;
    if (q1 && !q2 && s1 == s2) {
        for (int i = i0; i < i1; i++) {
            int x1 = s1 ? (int8_t)a[i] : a[i], x2 = s2 ? (int8_t)b[i] : b[i];
            x1 = clampi((x1 * m1 + a1) >> sh1, lo, hi);
            y[i] = (uint8_t)clampi(((x1 + x2) * om + oa) >> os, lo, hi);
        }
        return;
    }
    for (int i = i0; i < i1; i++) {
        int x1 = s1 ? (int8_t)a[i] : a[i], x2 = s2 ? (int8_t)b[i] : b[i];
        if (q1) x1 = clampi((x1 * m1 + a1) >> sh1, lo, hi);
        if (q2) x2 = clampi((x2 * m2 + a2) >> sh2, lo, hi);
        y[i] = (uint8_t)clampi(((x1 + x2) * om + oa) >> os, lo, hi);
    }
}

/* Where tensors and weights live: the platform's act_l2 / w_l2 point at L2 copies when it put them
 * there (small networks), else activations are in HyperRAM and weights in flash, staged through L2. */

/* cluster task phases: core 0 copies in, every core computes, core 0 copies out */
static struct {
    const uint8_t *src;        /* row y of the input is at src + y * W * C */
    int y0, H, W, C, pl;
    const uint8_t *wsrc, *klsrc;
    int wbytes, klbytes;
    uint8_t *dst;              /* output band (rows * Wo * cout) */
} X;

static void conv_in(void) {
    int C = X.C, rowb = J8.wp * C;
    for (int r = 0; r < J8.rin; r++) {
        uint8_t *d = J8.in1 + (size_t)r * rowb;
        int y = X.y0 + r;
        if (y < 0 || y >= X.H) { memset(d, 0, rowb); continue; }
        if (X.pl) memset(d, 0, (size_t)X.pl * C);
        if (J8.wp > X.pl + X.W) memset(d + (size_t)(X.pl + X.W) * C, 0, (size_t)(J8.wp - X.pl - X.W) * C);
        dma_in(d + (size_t)X.pl * C, X.src + (size_t)y * X.W * C, X.W * C);
    }
    dma_in(J8.w1, X.wsrc, X.wbytes);
    dma_in((uint8_t *)J8.kl, X.klsrc, X.klbytes);
}
static void tile_out(void) {   /* tile [rows*Wo][nco] -> band [rows*Wo][cout] at channel co0 */
    int px = J8.rows * J8.L->w;
    dma_out2d(X.dst + J8.co0, J8.out1, px * J8.nco, J8.cout, J8.nco);
}
static void flat_in(void) { dma_in(J8.in1, X.src, J8.n); dma_in(J8.in1 + J8.n, X.wsrc, J8.n); }
static void flat_out(void) { dma_out2d(X.dst, J8.out1, J8.n, J8.n, J8.n); }

static void timed_ram_rd(uint32_t off, void *dst, int n) { uint32_t t = now_us(); ram_rd(off, dst, n); io8_us += now_us() - t; }
static void timed_ram_wr(uint32_t off, const void *src, int n) { uint32_t t = now_us(); ram_wr(off, src, n); io8_us += now_us() - t; }
static void timed_fl_rd(uint32_t off, void *dst, int n) { uint32_t t = now_us(); fl_rd(off, dst, n); io8_us += now_us() - t; }
static void timed_ul_rd(uint32_t off, void *dst, int n) { uint32_t t = now_us(); ul_rd(off, dst, n); io8_us += now_us() - t; }

static void run_conv8(const Layer *L) {
    int dw = L->group != 1, k = L->k, s = L->s, Cin = L->cin, Cout = L->c, Ho = L->h, Wo = L->w, H = L->ih, W = L->iw;
    int wino = use_wino8 && L->uoff >= 0, cinp = (Cin + 1) & ~1;
    int pt = L->pads[0], pl = L->pads[1], wp = wino ? (Wo + 1) / 2 * 2 + 2 : W + pl + L->pads[3];
    int kp = dw ? k * k : wino ? 16 * cinp * 2 : L->kpad;   /* weight bytes per channel */
#define RIN(nr) (wino ? ((nr) + 1) / 2 * 2 + 2 : ((nr) - 1) * s + k)
    int col = wino ? NC8 * (2 * 16 * cinp * 2 + 512) : dw || k == 1 ? 0 : NC8 * 2 * kp;
    int nco = Cout, nr;
    if (!dw) while (nco > 4 && (nco * (kp + 8) > A1_n / 2 || AL4(RIN(1) * wp * Cin) + nco * (kp + 8) + Wo * nco + col > A1_n))
        nco = (nco / 2 + 3) / 4 * 4;
    int wfix = AL4(nco * kp) + nco * 8;
#define L1NEED(nr) (AL4(RIN(nr) * wp * Cin) + wfix + AL4((nr) * Wo * nco) + col)
    for (nr = Ho; nr > 1 && (L1NEED(nr) > A1_n || (wino && (nr & 1) && nr != Ho)); nr--);
    if (L1NEED(nr) > A1_n) FAIL8("conv tile does not fit L1");
    /* L2 staging: input rows and output band unless activations are in L2; weights unless they are */
    int in_b = act_l2 ? 0 : AL4(RIN(nr) * W * Cin), out_b = act_l2 ? 0 : AL4(nr * Wo * Cout);
    int wall = w_l2 ? 0 : AL4(Cout * kp) + Cout * 8;
    int resident = w_l2 || in_b + out_b + wall <= A2_n;
    if (!resident && (dw || in_b + out_b + AL4(nco * kp) + nco * 8 > A2_n)) FAIL8("conv band does not fit L2");
    uint8_t *in2 = A2, *out2 = A2 + in_b, *w2 = out2 + out_b;
    const uint8_t *wsrc_l2 = wino ? u_l2 : w_l2;
    int woff = wino ? L->uoff : L->woff;
    void (*wrd)(uint32_t, void *, int) = wino ? timed_ul_rd : timed_fl_rd;
    if (wino) resident = u_l2 || in_b + out_b + AL4(Cout * kp) + Cout * 8 <= A2_n;
    const uint8_t *wbase = wsrc_l2 ? wsrc_l2 + woff : w2, *klbase = w_l2 ? w_l2 + L->koff : w2 + AL4(Cout * kp);
    if (!wsrc_l2 && resident) wrd(woff, w2, Cout * kp);
    if (!w_l2 && resident) timed_fl_rd(L->koff, w2 + AL4(Cout * kp), Cout * 8);
    J8.L = L; J8.wp = wp; J8.cout = Cout;
    J8.in1 = A1; J8.w1 = A1 + AL4(RIN(nr) * wp * Cin); J8.kl = (const int32_t *)(J8.w1 + AL4(nco * kp));
    J8.out1 = J8.w1 + wfix; J8.col1 = J8.out1 + AL4(nr * Wo * nco);
    X.H = H; X.W = W; X.C = Cin; X.pl = pl;
    for (int y0 = 0; y0 < Ho; y0 += nr) {
        int rows = Ho - y0 < nr ? Ho - y0 : nr, yin = y0 * s - pt, rin = RIN(rows);
        X.y0 = yin;
        if (act_l2) X.src = act_l2 + L->in;
        else {     /* stage the band's valid rows */
            int ya = yin < 0 ? 0 : yin, yb = yin + rin > H ? H : yin + rin;
            timed_ram_rd(L->in + (uint32_t)ya * W * Cin, in2, (yb - ya) * W * Cin);
            X.src = in2 - (size_t)ya * W * Cin;
        }
        X.dst = act_l2 ? act_l2 + L->out + (size_t)y0 * Wo * Cout : out2;
        for (int co0 = 0; co0 < Cout; co0 += nco) {
            int n = Cout - co0 < nco ? Cout - co0 : nco;
            if (resident) { X.wsrc = wbase + (size_t)co0 * kp; X.klsrc = klbase + co0 * 8; }
            else {
                wrd(woff + co0 * kp, w2, n * kp);
                timed_fl_rd(L->koff + co0 * 8, w2 + AL4(nco * kp), n * 8);
                X.wsrc = w2; X.klsrc = w2 + AL4(nco * kp);
            }
            X.wbytes = n * kp; X.klbytes = n * 8;
            J8.rows = rows; J8.rin = rin; J8.co0 = co0; J8.nco = n;
            cl_run(conv_in, dw ? k_dw : wino ? k_wino8 : k_conv, tile_out);
        }
        if (!act_l2) timed_ram_wr(L->out + (uint32_t)y0 * Wo * Cout, out2, rows * Wo * Cout);
    }
#undef RIN
#undef L1NEED
}

static void run_add8(const Layer *L) {
    int N = L->c * L->h * L->w, chunk = (A1_n / 3) & ~3;
    if (!act_l2 && chunk > A2_n / 3) chunk = (A2_n / 3) & ~3;
    J8.L = L;
    for (int i0 = 0; i0 < N; i0 += chunk) {
        int n = N - i0 < chunk ? N - i0 : chunk;
        if (act_l2) { X.src = act_l2 + L->in + i0; X.wsrc = act_l2 + L->in2 + i0; X.dst = act_l2 + L->out + i0; }
        else {
            timed_ram_rd(L->in + i0, A2, n);
            timed_ram_rd(L->in2 + i0, A2 + n, n);
            X.src = A2; X.wsrc = A2 + n; X.dst = A2 + 2 * n;
        }
        J8.n = n; J8.in1 = A1; J8.out1 = A1 + 2 * n;
        cl_run(flat_in, k_add, flat_out);
        if (!act_l2) timed_ram_wr(L->out + i0, A2 + 2 * n, n);
    }
}

/* a tensor's bytes, read in place when activations are in L2 */
static const uint8_t *act_read(uint32_t off, int n, uint8_t *stage) {
    if (act_l2) return act_l2 + off;
    timed_ram_rd(off, stage, n);
    return stage;
}

/* global average pool + requant, and the final fully connected layer: small, run on the FC */
static void run_gap8(const Layer *L) {
    int C = L->c, n = L->p[2], sgn = L->in_signed;
    const uint8_t *x = act_read(L->in, C * n, A2);
    uint8_t *y = A2 + C * n;
    for (int c = 0; c < C; c++) {
        int32_t sum = 0;
        for (int i = 0; i < n; i++) sum += sgn ? (int8_t)x[i * C + c] : x[i * C + c];
        /* floor((sum / n * m + a) / 2^d) exactly: (sum * m + a * n) / (n * 2^d), floored */
        int64_t num = (int64_t)sum * L->p[0] + (int64_t)L->p[1] * n, den = (int64_t)n << L->d;
        int64_t q = num >= 0 ? num / den : -((-num + den - 1) / den);
        y[c] = (uint8_t)clampi((int)q, L->lo, L->hi);
    }
    timed_ram_wr(L->out, y, C);
}

static void run_fc8(const Layer *L) {
    int nin = L->cin, nout = L->c, sgn = L->in_signed;
    const uint8_t *x = act_read(L->in, nin, A2);
    int32_t *y = (int32_t *)(A2 + (nin + 3) / 4 * 4), *b = y + nout;
    int8_t *w = (int8_t *)(b + nout);
    int rows = (A2_n - (int)((uint8_t *)w - A2)) / nin;
    if (rows < 1) FAIL8("fc row does not fit L2");
    timed_fl_rd(L->koff, b, nout * 4);
    for (int r0 = 0; r0 < nout; r0 += rows) {
        int n = nout - r0 < rows ? nout - r0 : rows;
        timed_fl_rd(L->woff + r0 * nin, w, n * nin);
        for (int r = 0; r < n; r++) {
            int32_t s = b[r0 + r];
            for (int i = 0; i < nin; i++) s += (sgn ? (int8_t)x[i] : x[i]) * w[(size_t)r * nin + i];
            y[r0 + r] = s;
        }
    }
    timed_ram_wr(L->out, y, nout * 4);
}

static void run_net8(void) {
    memset(cat8_us, 0, sizeof cat8_us);
    io8_us = 0;
    for (int i = 0; i < N_LAYERS; i++) {
        const Layer *L = &LAYERS[i];
        uint32_t t0 = now_us();
        int cat = L->type == L_CONV ? (L->group != 1 ? C8_DW : C8_CONV) : L->type == L_ADD ? C8_ADD : L->type == L_GAP ? C8_GAP : C8_FC;
        if (L->type == L_CONV) run_conv8(L);
        else if (L->type == L_ADD) run_add8(L);
        else if (L->type == L_GAP) run_gap8(L);
        else run_fc8(L);
        lay_us[i] = now_us() - t0;
        cat8_us[cat] += lay_us[i];
        if (fail8) return;
        if (after_layer8) after_layer8(i);
    }
}
