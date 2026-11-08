"""Convert MLPerf Tiny's int8 ResNet-8 (TFLite) into a Quantlab-style integer ONNX, the form DORY
reads and qref.py / our GAP8 engine run, plus DORY's golden files.

    python tflite2ql.py MODEL.tflite CIFAR_DIR OUT_DIR

Every activation is kept as u = q + 128 (unsigned, zero point zp + 128), so nothing is signed:
  conv:  u_out = floor((sum(u*w) * k + l) / 2^d), clipped to 0..255, with k ~ M * 2^d and the bias,
         rounding and output zero point folded into l. d is chosen per layer so sum*k stays far
         inside int32 (DORY multiplies in 32 bits), from the largest sums over calibration images.
  add:   the smaller-scale input is rescaled to the other's scale (rq), then the sum requantized.
  pool:  GlobalAveragePool, then floor(mean * 2^19 + 2^18) / 2^19, i.e. round half up.
  fc:    int32 logits = W u + b (argmax = the TFLite model's).
TFLite rounds with its own fixed-point multiplier, so outputs can differ from TFLite by 1 in rare
places; qref.py is the exact reference for DORY and our engine.
"""
import os
import pickle
import sys

import numpy as np
import onnx
import tflite
from onnx import TensorProto, helper, numpy_helper

import qref

N_CALIB = 200     # images used to bound the conv sums
MARGIN = 4.0      # headroom over the largest sum seen


def tensors(model, g):
    def t(i):
        x = g.Tensors(i)
        q = x.Quantization()
        buf = model.Buffers(x.Buffer()).DataAsNumpy()
        dt = {9: np.int8, 2: np.int32}.get(x.Type())
        data = None if isinstance(buf, int) or dt is None else np.frombuffer(buf.tobytes(), dt).reshape(x.ShapeAsNumpy())
        return dict(shape=x.ShapeAsNumpy().tolist(), scale=q.ScaleAsNumpy() if q.ScaleLength() else None,
                    zp=int(q.ZeroPointAsNumpy()[0]) if q.ZeroPointLength() else 0, data=data)
    return t


def load_tflite(path):
    model = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
    g = model.Subgraphs(0)
    t = tensors(model, g)
    names = {getattr(tflite.BuiltinOperator, k): k for k in dir(tflite.BuiltinOperator) if not k.startswith("_")}
    ops = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        kind = names[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        rec = dict(kind=kind, ins=ins, out=op.Outputs(0), t=t)
        if kind == "CONV_2D":
            o = tflite.Conv2DOptions()
            o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            rec.update(stride=o.StrideH(), same=o.Padding() == tflite.Padding.SAME)
        ops.append(rec)
    return ops, g.Inputs(0), t


def build(ops, t, d_of, Ms=None):
    """The Quantlab-style graph. d_of(i) gives conv i's shift; Ms collects each conv's largest multiplier."""
    nodes, inits, n = [], [], [100]
    def name(p):       # tensors are numbered, as in a PyTorch export (DORY sorts nodes by these numbers)
        n[0] += 1
        return str(n[0])
    def init(arr, p):
        n[0] += 1   # DORY tells weights from biases by name
        nm = f"layer{n[0]}." + {"w": "weight", "k": "mul", "l": "add", "fcw": "weight", "fcb": "bias"}.get(p, p)
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), nm))
        return nm
    def const(v):
        out = name("c")
        nodes.append(helper.make_node("Constant", [], [out], value=numpy_helper.from_array(np.array(v, np.float32))))
        return out
    def requant(x, k, l, d, lo=0.0, hi=255.0):
        m = name("mul"); a = name("add"); dv = name("div"); f = name("floor"); c = name("clip")
        nodes.append(helper.make_node("Mul", [x, init(k, "k")], [m], mult_bits=32))
        nodes.append(helper.make_node("Add", [m, init(l, "l")], [a], add_bits=32))
        nodes.append(helper.make_node("Div", [a, const(2.0 ** d)], [dv]))
        nodes.append(helper.make_node("Floor", [dv], [f]))
        nodes.append(helper.make_node("Clip", [f], [c], min=lo, max=hi, out_bits=8))
        return c

    names, final = {}, [None]
    first = ops[0]["ins"][0]
    names[first] = "input.1"
    conv_i = 0
    for op in ops:
        kind = op["kind"]
        if kind == "CONV_2D":
            x, w, b = (t(i) for i in op["ins"])
            y = t(op["out"])
            assert x["zp"] == -128, "conv inputs are post-ReLU (zero point -128), so padding with 0 is exact"
            W = w["data"].transpose(0, 3, 1, 2)            # OHWI -> OIHW
            M = x["scale"][0] * w["scale"] / y["scale"][0]
            d = d_of(conv_i)
            if Ms is not None:
                Ms.append(float(M.max()))
            k = np.round(M * 2.0 ** d)
            l = np.round(b["data"] * M * 2.0 ** d) + (y["zp"] + 128) * 2.0 ** d + 2.0 ** (d - 1)
            kk = W.shape[2]
            p = (kk - 1) // 2 if op["same"] else 0
            if op["same"] and op["stride"] == 2:
                p = [0, 0, 1, 1] if kk == 3 else [0, 0, 0, 0]   # TFLite SAME at stride 2 pads bottom/right only
            pads = p if isinstance(p, list) else [p] * 4
            cv = name("conv")
            nodes.append(helper.make_node("Conv", [names[op["ins"][0]], init(W, "w")], [cv], dilations=[1, 1], group=1,
                                          kernel_shape=[kk, kk], pads=pads, strides=[op["stride"]] * 2,
                                          weight_bits=8, bias_bits=32))
            names[op["out"]] = requant(cv, k.reshape(-1, 1, 1), l.reshape(-1, 1, 1), d)
            conv_i += 1
        elif kind == "ADD":
            a_, b_ = (t(i) for i in op["ins"])
            y = t(op["out"])
            ia, ib = op["ins"]
            if a_["scale"][0] > b_["scale"][0]:          # rescale the smaller-scale input onto the other
                a_, b_, ia, ib = b_, a_, ib, ia
            r = a_["scale"][0] / b_["scale"][0]
            za, zb, zo = a_["zp"] + 128, b_["zp"] + 128, y["zp"] + 128
            s1, so = 19, 19
            c1 = round(za * r)
            m1 = round(r * 2 ** s1)
            a1 = round((-za * r + c1 + 0.5) * 2 ** s1)
            ro = b_["scale"][0] / y["scale"][0]
            om = round(ro * 2 ** so)
            oa = round((-(c1 + zb) * ro + zo + 0.5) * 2 ** so)
            out = name("addq")
            nodes.append(helper.make_node("Add", [names[ia], names[ib]], [out], in1_mul=m1, in1_add=a1, in1_shift=s1,
                                          in1_rq=1, in1_n_levels=256, in2_mul=1, in2_add=0, in2_shift=0, in2_rq=0,
                                          in2_n_levels=256, out_mul=om, out_add=oa, out_shift=so, out_rq=1,
                                          out_n_levels=256, add_bits=8))
            names[op["out"]] = out
        elif kind == "AVERAGE_POOL_2D":
            gp = name("gap")
            nodes.append(helper.make_node("GlobalAveragePool", [names[op["ins"][0]]], [gp]))
            names[op["out"]] = requant(gp, np.array(2.0 ** 19), np.array(2.0 ** 18), 19)
        elif kind == "RESHAPE":
            rs = name("reshape")
            nodes.append(helper.make_node("Reshape", [names[op["ins"][0]], init(np.array([1, -1]), "shape")], [rs]))
            names[op["out"]] = rs
        elif kind == "FULLY_CONNECTED":
            x, w, b = (t(i) for i in op["ins"])
            assert x["zp"] == -128
            out = name("fc")
            final[0] = out
            nodes.append(helper.make_node("Gemm", [names[op["ins"][0]], init(w["data"], "fcw"), init(b["data"], "fcb")],
                                          [out], alpha=1.0, beta=1.0, transB=1, weight_bits=8, bias_bits=32))
            names[op["out"]] = out
        elif kind == "SOFTMAX":
            pass     # argmax of the logits
        else:
            raise ValueError(kind)
    g = helper.make_graph(nodes, "resnet8_mlperf_int8", [helper.make_tensor_value_info("input.1", TensorProto.FLOAT, [1, 3, 32, 32])],
                          [helper.make_tensor_value_info(final[0], TensorProto.FLOAT, [1, 10])], inits)
    m = helper.make_model(g, producer_name="pytorch", opset_imports=[helper.make_opsetid("", 10)])
    m.ir_version = 6
    return m


def cifar(cifar_dir):
    d = pickle.load(open(os.path.join(cifar_dir, "cifar-10-batches-py", "test_batch"), "rb"), encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).astype(np.float64), np.array(d[b"labels"])


def write_golden(path, title, arr):
    with open(path, "w") as f:
        f.write(f"# {title} (shape {list(arr.shape)}),\n")
        f.write("".join(f"{int(v)},\n" for v in arr.ravel()))


def main():
    tfl, cifar_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    ops, _, t = load_tflite(tfl)
    x, y = cifar(cifar_dir)
    # pass 1 with a small shift, to measure each conv's largest |sum| (the conv output before Mul)
    Ms = []
    probe = build(ops, t, lambda i: 8, Ms)
    convs = [nd.output[0] for nd in probe.graph.node if nd.op_type == "Conv"]
    accmax = {c: 0.0 for c in convs}
    for img in x[:N_CALIB]:
        T, _ = qref.run(probe, img)
        for c in convs:
            accmax[c] = max(accmax[c], np.abs(T[c]).max())
    # largest d with MARGIN * accmax * M * 2^d < 2^31
    ds = [min(24, int(np.floor(np.log2(2.0 ** 31 / (MARGIN * accmax[c] * m))))) for c, m in zip(convs, Ms)]
    print("largest conv sums", [int(accmax[c]) for c in convs])
    print("per-conv shift d:", ds)
    model = build(ops, t, lambda i: ds[i])
    onnx.save(model, os.path.join(out, "model_int.onnx"))
    # accuracy of the integer model, and DORY's golden files for the first test image
    correct = 0
    N = 1000
    for i in range(N):
        T, ends = qref.run(model, x[i])
        correct += int(np.argmax(T[model.graph.output[0].name]) == y[i])
    print(f"integer model: {correct}/{N} correct on the CIFAR-10 test set")
    T, ends = qref.run(model, x[0])
    write_golden(os.path.join(out, "input.txt"), "input", x[0].transpose(1, 2, 0))
    for i, e in enumerate(ends):
        v = T[e]
        write_golden(os.path.join(out, f"out_layer{i}.txt"), e, v.transpose(1, 2, 0) if v.ndim == 3 else v)
    write_golden(os.path.join(out, "output.txt"), "output", T[model.graph.output[0].name])
    print(len(ends), "layers, golden files written; first image logits", T[model.graph.output[0].name].astype(int).tolist(), "label", y[0])


if __name__ == "__main__":
    main()
