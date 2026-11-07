/*
 * src/xnnpack_runner.c
 *
 * Runs a model plan from benchmarks/whole_model.py on XNNPACK's subgraph API, so it
 * uses the same graph, weights and input as ONNX Runtime and src/model_runner.c.
 * XNNPACK works in NHWC, so weights and the input are reordered during setup, and
 * flattening goes through a transpose so its output order matches ONNX.
 */
#define main harness_main
#include "bench_harness.c"
#undef main

#include <xnnpack.h>
#include <pthreadpool.h>

#define MAX_T 256
#define MAX_LINES 1024

typedef struct { char op[16]; int a[12]; long l[2]; float f[2]; } Line;
typedef struct { int c, h, w; uint32_t value; int defined; } Tn;

static Line g_lines[MAX_LINES];
static int g_n;
static Tn g_tn[MAX_T];
static xnn_subgraph_t g_sg;
#define MAX_OUTPUTS 8
static int g_outs[MAX_OUTPUTS];
static int g_n_outs;

static void die(const char* msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(2);
}

static void check(enum xnn_status s, const char* what) {
    if (s != xnn_status_success) {
        fprintf(stderr, "FATAL: XNNPACK %s failed (status %d)\n", what, (int)s);
        exit(3);
    }
}

static uint32_t new_value(size_t n, const size_t* dims, const void* data, uint32_t ext, uint32_t flags) {
    uint32_t id = XNN_INVALID_VALUE_ID;
    check(xnn_define_tensor_value(g_sg, xnn_datatype_fp32, n, dims, data, ext, flags, &id), "define tensor");
    return id;
}

static float* read_floats(const char* path, size_t* count, size_t extra) {
    FILE* f = fopen(path, "rb");
    if (!f) die("cannot open float file");
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    float* d = (float*)malloc(bytes + extra);
    if (!d || fread(d, 1, bytes, f) != bytes) die("short read");
    fclose(f);
    if (count) *count = bytes / sizeof(float);
    return d;
}

static void read_plan(const char* path, int* in_id) {
    FILE* f = fopen(path, "r");
    if (!f) die("cannot open plan");
    char w[16];
    while (fscanf(f, "%15s", w) == 1) {
        if (strcmp(w, "TENSOR") == 0) {
            int id, c, h, wd;
            if (fscanf(f, "%d %d %d %d", &id, &c, &h, &wd) != 4 || id >= MAX_T) die("bad TENSOR");
            g_tn[id] = (Tn){c, h, wd, XNN_INVALID_VALUE_ID, 0};
        } else if (strcmp(w, "INPUT") == 0) {
            if (fscanf(f, "%d", in_id) != 1) die("bad INPUT");
        } else if (strcmp(w, "OUTPUT") == 0) {
            if (g_n_outs >= MAX_OUTPUTS || fscanf(f, "%d", &g_outs[g_n_outs++]) != 1) die("bad OUTPUT");
        } else {
            Line* L = &g_lines[g_n++];
            memset(L, 0, sizeof(*L));
            strcpy(L->op, w);
            int ok;
            if (strcmp(w, "CONV") == 0) {
                ok = fscanf(f, "%d %d %d %d %d %d %d %d %ld %ld", &L->a[0], &L->a[1], &L->a[2], &L->a[3], &L->a[4],
                            &L->a[5], &L->a[6], &L->a[7], &L->l[0], &L->l[1]) == 10;
            } else if (strcmp(w, "DWCONV") == 0) {
                ok = fscanf(f, "%d %d %d %d %d %d %ld %ld", &L->a[0], &L->a[1], &L->a[2], &L->a[4], &L->a[5],
                            &L->a[6], &L->l[0], &L->l[1]) == 8;
            } else if (strcmp(w, "CLIP") == 0) {
                ok = fscanf(f, "%d %d %f %f", &L->a[0], &L->a[1], &L->f[0], &L->f[1]) == 4;
            } else if (strcmp(w, "SPLIT") == 0) {
                /* a[1] stays the input, as for every other op */
                ok = fscanf(f, "%d %d %d", &L->a[0], &L->a[2], &L->a[1]) == 3;
            } else if (strcmp(w, "CONCAT") == 0) {
                /* a[2] is the count; inputs are a[1] then a[3] onward */
                ok = fscanf(f, "%d %d %d", &L->a[0], &L->a[2], &L->a[1]) == 3 && L->a[2] <= 8;
                for (int j = 3; ok && j < L->a[2] + 2; j++) ok = fscanf(f, "%d", &L->a[j]) == 1;
            } else if (strcmp(w, "RELU") == 0 || strcmp(w, "GAP") == 0 || strcmp(w, "SILU") == 0 ||
                       strcmp(w, "UPSAMPLE") == 0) {
                ok = fscanf(f, "%d %d", &L->a[0], &L->a[1]) == 2;
            } else if (strcmp(w, "MAXPOOL") == 0) {
                ok = fscanf(f, "%d %d %d %d %d", &L->a[0], &L->a[1], &L->a[2], &L->a[3], &L->a[4]) == 5;
            } else if (strcmp(w, "ADD") == 0) {
                ok = fscanf(f, "%d %d %d", &L->a[0], &L->a[1], &L->a[2]) == 3;
            } else if (strcmp(w, "GEMM") == 0) {
                ok = fscanf(f, "%d %d %d %d %ld %ld", &L->a[0], &L->a[1], &L->a[2], &L->a[3], &L->l[0], &L->l[1]) == 6;
            } else {
                ok = 0;
            }
            if (!ok) die("bad plan line");
        }
    }
    fclose(f);
}

/* Value for one op's output; the last write to a plan output tensor is external. */
static uint32_t output_value(int plan_id, int line, int c, int h, int w) {
    int is_output = 0;
    uint32_t ext = XNN_INVALID_VALUE_ID;
    for (int k = 0; k < g_n_outs; k++) {
        if (g_outs[k] == plan_id) {
            is_output = 1;
            ext = 1 + (uint32_t)k;
        }
    }
    for (int j = line + 1; is_output && j < g_n; j++) {
        if (g_lines[j].a[0] == plan_id) is_output = 0;
    }
    size_t dims[4] = {1, (size_t)h, (size_t)w, (size_t)c};
    uint32_t id = new_value(4, dims, NULL, is_output ? ext : XNN_INVALID_VALUE_ID,
                            is_output ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0);
    g_tn[plan_id].value = id;
    return id;
}

int main(int argc, char** argv) {
    const char* plan = NULL, *weights_path = NULL, *input_path = NULL, *output_path = NULL, *csv = NULL;
    int T = 4, warmup = 10, runs = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--plan") == 0) plan = argv[i + 1];
        else if (strcmp(argv[i], "--weights") == 0) weights_path = argv[i + 1];
        else if (strcmp(argv[i], "--input") == 0) input_path = argv[i + 1];
        else if (strcmp(argv[i], "--output") == 0) output_path = argv[i + 1];
        else if (strcmp(argv[i], "--csv") == 0) csv = argv[i + 1];
        else if (strcmp(argv[i], "--T") == 0) T = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--warmup") == 0) warmup = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--runs") == 0) runs = atoi(argv[i + 1]);
        else die("unknown argument");
    }
    if (!plan || !weights_path || !input_path || !output_path || !csv) die("missing argument");

    check(xnn_initialize(NULL), "initialize");
    int in_id = -1;
    read_plan(plan, &in_id);
    if (in_id < 0 || g_n_outs == 0) die("plan has no INPUT or OUTPUT");
    float* weights = read_floats(weights_path, NULL, 0);

    const uint32_t EXT_IN = 0;
    check(xnn_create_subgraph(1 + g_n_outs, 0, &g_sg), "create subgraph");

    /* Input: CHW file to the HWC layout XNNPACK takes */
    Tn* ti = &g_tn[in_id];
    size_t n_in = 0;
    float* chw = read_floats(input_path, &n_in, 0);
    if (n_in != (size_t)ti->c * ti->h * ti->w) die("input size mismatch");
    float* hwc = (float*)malloc(n_in * sizeof(float) + XNN_EXTRA_BYTES);
    for (int c = 0; c < ti->c; c++) {
        for (int h = 0; h < ti->h; h++) {
            for (int w = 0; w < ti->w; w++) {
                hwc[((size_t)h * ti->w + w) * ti->c + c] = chw[((size_t)c * ti->h + h) * ti->w + w];
            }
        }
    }
    size_t in_dims[4] = {1, (size_t)ti->h, (size_t)ti->w, (size_t)ti->c};
    ti->value = new_value(4, in_dims, NULL, EXT_IN, XNN_VALUE_FLAG_EXTERNAL_INPUT);

    const float INF = INFINITY;
    for (int i = 0; i < g_n; i++) {
        Line* L = &g_lines[i];
        int out = L->a[0];
        Tn* src = &g_tn[L->a[1]];
        Tn* dst = &g_tn[out];
        uint32_t in_v = src->value;
        if (in_v == XNN_INVALID_VALUE_ID) die("op input has no value");

        if (strcmp(L->op, "CONV") == 0) {
            int cin = L->a[2], cout = L->a[3], k = L->a[4], s = L->a[5], p = L->a[6];
            /* ONNX [Cout, Cin, k, k] to XNNPACK [Cout, k, k, Cin] */
            float* filter = (float*)malloc((size_t)cout * cin * k * k * sizeof(float));
            const float* W = weights + L->l[0];
            for (int co = 0; co < cout; co++) {
                for (int ci = 0; ci < cin; ci++) {
                    for (int t = 0; t < k * k; t++) {
                        filter[(((size_t)co * k * k) + t) * cin + ci] = W[((size_t)co * cin + ci) * k * k + t];
                    }
                }
            }
            size_t fd[4] = {(size_t)cout, (size_t)k, (size_t)k, (size_t)cin};
            uint32_t f = new_value(4, fd, filter, XNN_INVALID_VALUE_ID, 0);
            uint32_t b = XNN_INVALID_VALUE_ID;
            if (L->l[1] >= 0) {
                size_t bd[1] = {(size_t)cout};
                b = new_value(1, bd, weights + L->l[1], XNN_INVALID_VALUE_ID, 0);
            }
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_convolution_2d(g_sg, p, p, p, p, k, k, s, s, 1, 1, 1, cin, cout, -INF, INF,
                                            in_v, f, b, o, 0), "define conv");
        } else if (strcmp(L->op, "DWCONV") == 0) {
            int c = L->a[2], k = L->a[4], s = L->a[5], p = L->a[6];
            /* ONNX [C, 1, k, k] to XNNPACK [1, k, k, C] */
            float* filter = (float*)malloc((size_t)c * k * k * sizeof(float));
            const float* W = weights + L->l[0];
            for (int ci = 0; ci < c; ci++) {
                for (int t = 0; t < k * k; t++) filter[(size_t)t * c + ci] = W[(size_t)ci * k * k + t];
            }
            size_t fd[4] = {1, (size_t)k, (size_t)k, (size_t)c};
            uint32_t f = new_value(4, fd, filter, XNN_INVALID_VALUE_ID, 0);
            uint32_t b = XNN_INVALID_VALUE_ID;
            if (L->l[1] >= 0) {
                size_t bd[1] = {(size_t)c};
                b = new_value(1, bd, weights + L->l[1], XNN_INVALID_VALUE_ID, 0);
            }
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_depthwise_convolution_2d(g_sg, p, p, p, p, k, k, s, s, 1, 1, 1, c, -INF, INF,
                                                      in_v, f, b, o, 0), "define depthwise conv");
        } else if (strcmp(L->op, "CLIP") == 0) {
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            union xnn_unary_params params;
            params.clamp.min = L->f[0];
            params.clamp.max = L->f[1];
            check(xnn_define_unary(g_sg, xnn_unary_clamp, &params, in_v, o, 0), "define clip");
        } else if (strcmp(L->op, "SILU") == 0) {
            size_t d[4] = {1, (size_t)src->h, (size_t)src->w, (size_t)src->c};
            uint32_t sig = new_value(4, d, NULL, XNN_INVALID_VALUE_ID, 0);
            check(xnn_define_unary(g_sg, xnn_unary_sigmoid, NULL, in_v, sig, 0), "define sigmoid");
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_binary(g_sg, xnn_binary_multiply, NULL, in_v, sig, o, 0), "define multiply");
        } else if (strcmp(L->op, "SPLIT") == 0) {
            Tn* d2 = &g_tn[L->a[2]];
            size_t off0[4] = {0, 0, 0, 0}, off1[4] = {0, 0, 0, (size_t)dst->c};
            size_t size[4] = {1, (size_t)dst->h, (size_t)dst->w, (size_t)dst->c};
            uint32_t o0 = output_value(out, i, dst->c, dst->h, dst->w);
            uint32_t o1 = output_value(L->a[2], i, d2->c, d2->h, d2->w);
            check(xnn_define_static_slice(g_sg, 4, off0, size, in_v, o0, 0), "define slice");
            check(xnn_define_static_slice(g_sg, 4, off1, size, in_v, o1, 0), "define slice");
        } else if (strcmp(L->op, "CONCAT") == 0) {
            uint32_t ids[8];
            ids[0] = in_v;
            for (int j = 1; j < L->a[2]; j++) ids[j] = g_tn[L->a[j + 2]].value;
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_concatenate(g_sg, 3, (size_t)L->a[2], ids, o, 0), "define concat");
        } else if (strcmp(L->op, "UPSAMPLE") == 0) {
            /* XNNPACK has no nearest resize, so use a fixed depthwise deconvolution of ones,
             * which gives the same pixels. This may cost more than a real nearest resize. */
            int c = src->c;
            float* ones = (float*)malloc((size_t)c * 4 * sizeof(float));
            for (int j = 0; j < c * 4; j++) ones[j] = 1.0f;
            size_t fd[4] = {(size_t)c, 2, 2, 1};
            uint32_t f = new_value(4, fd, ones, XNN_INVALID_VALUE_ID, 0);
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_deconvolution_2d(g_sg, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 1, c, 1, 1, -INF, INF,
                                              in_v, f, XNN_INVALID_VALUE_ID, o, 0), "define upsample");
        } else if (strcmp(L->op, "RELU") == 0) {
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            union xnn_unary_params params;
            params.clamp.min = 0.0f;
            params.clamp.max = INF;
            check(xnn_define_unary(g_sg, xnn_unary_clamp, &params, in_v, o, 0), "define relu");
        } else if (strcmp(L->op, "MAXPOOL") == 0) {
            int k = L->a[2], s = L->a[3], p = L->a[4];
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_max_pooling_2d(g_sg, p, p, p, p, k, k, s, s, 1, 1, -INF, INF, in_v, o, 0), "define maxpool");
        } else if (strcmp(L->op, "ADD") == 0) {
            uint32_t b = g_tn[L->a[2]].value;
            if (b == XNN_INVALID_VALUE_ID) die("add input has no value");
            uint32_t o = output_value(out, i, dst->c, dst->h, dst->w);
            check(xnn_define_binary(g_sg, xnn_binary_add, NULL, in_v, b, o, 0), "define add");
        } else if (strcmp(L->op, "GAP") == 0) {
            const size_t axes[2] = {1, 2};
            uint32_t o = output_value(out, i, dst->c, 1, 1);
            check(xnn_define_static_reduce(g_sg, xnn_reduce_mean, 2, axes, in_v, o, XNN_FLAG_KEEP_DIMS), "define mean");
        } else if (strcmp(L->op, "GEMM") == 0) {
            int nin = L->a[2], nout = L->a[3];
            if (src->h * src->w > 1) {
                /* Flatten in ONNX's CHW order */
                const size_t perm[4] = {0, 3, 1, 2};
                size_t td[4] = {1, (size_t)src->c, (size_t)src->h, (size_t)src->w};
                uint32_t t = new_value(4, td, NULL, XNN_INVALID_VALUE_ID, 0);
                check(xnn_define_static_transpose(g_sg, 4, perm, in_v, t, 0), "define transpose");
                size_t rd[4] = {1, 1, 1, (size_t)nin};
                uint32_t r = new_value(4, rd, NULL, XNN_INVALID_VALUE_ID, 0);
                check(xnn_define_static_reshape(g_sg, 4, rd, t, r, 0), "define reshape");
                in_v = r;
            }
            size_t fd[2] = {(size_t)nout, (size_t)nin};
            uint32_t f = new_value(2, fd, weights + L->l[0], XNN_INVALID_VALUE_ID, 0);
            uint32_t b = XNN_INVALID_VALUE_ID;
            if (L->l[1] >= 0) {
                size_t bd[1] = {(size_t)nout};
                b = new_value(1, bd, weights + L->l[1], XNN_INVALID_VALUE_ID, 0);
            }
            uint32_t o = output_value(out, i, nout, 1, 1);
            check(xnn_define_fully_connected(g_sg, -INF, INF, in_v, f, b, o, 0), "define fc");
        } else {
            die("unknown op");
        }
    }

    float* outputs[MAX_OUTPUTS];
    struct xnn_external_value ext[1 + MAX_OUTPUTS] = {{EXT_IN, hwc}};
    for (int k = 0; k < g_n_outs; k++) {
        Tn* to = &g_tn[g_outs[k]];
        outputs[k] = (float*)malloc((size_t)to->c * to->h * to->w * sizeof(float) + XNN_EXTRA_BYTES);
        ext[1 + k] = (struct xnn_external_value){1 + (uint32_t)k, outputs[k]};
    }

    pthreadpool_t pool = pthreadpool_create((size_t)T);
    xnn_runtime_t rt = NULL;
    check(xnn_create_runtime_v2(g_sg, pool, 0, &rt), "create runtime");
    check(xnn_reshape_runtime(rt), "reshape runtime");
    check(xnn_setup_runtime_v2(rt, 1 + g_n_outs, ext), "setup runtime");

    for (int r = 0; r < warmup; r++) check(xnn_invoke_runtime(rt), "invoke");

    FILE* fp = fopen(csv, "w");
    if (!fp) die("cannot open csv");
    fprintf(fp, "run_id,total_ms,freq_mhz,temp_c,throttle_hex\n");
    for (int r = 0; r < runs; r++) {
        long freq = get_cpu_freq_mhz();
        double temp = get_cpu_temp_c();
        unsigned int throttle = get_throttle_flags();
        double t0 = get_time_ms();
        check(xnn_invoke_runtime(rt), "invoke");
        fprintf(fp, "%d,%.6f,%ld,%.2f,0x%x\n", r, get_time_ms() - t0, freq, temp, throttle);
    }
    fclose(fp);

    FILE* fo = fopen(output_path, "wb");
    if (!fo) die("cannot open output");
    for (int k = 0; k < g_n_outs; k++) {
        /* HWC back to ONNX's CHW; fully connected outputs are flat, so this leaves them alone */
        Tn* to = &g_tn[g_outs[k]];
        for (int c = 0; c < to->c; c++) {
            for (size_t px = 0; px < (size_t)to->h * to->w; px++) {
                fwrite(&outputs[k][px * to->c + c], sizeof(float), 1, fo);
            }
        }
    }
    fclose(fo);
    return 0;
}
