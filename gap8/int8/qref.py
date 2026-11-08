"""Integer reference for DORY/Quantlab-style 8-bit networks (exact: float64 holds every value).

    python qref.py MODEL_INT.onnx GOLDEN_DIR   # checks every layer against DORY's out_layerN.txt

Semantics (Quantlab "integerized" ONNX): activations uint8, weights int8, int32 sums.
  conv layer: y = clip(floor((sum(x*w) * k + l) / 2^d), 0, 255)            (Conv-Mul-Add-Div-Floor-Clip)
  residual add: y = clip(floor((rq1(a) + rq2(b)) * om + oa) / 2^os)), rq(x) = floor((x*m + a) / 2^s) if rq,
                clipped to int8 when an input is signed (projection outputs are), else uint8
  GlobalAveragePool, then the final Gemm with an int32 output.
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper


def conv(x, w, stride, pad, group):
    """x [C,H,W], w [O,C/g,k,k], exact in float64"""
    c, h, wd = x.shape
    o, cg, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3])))
    ho = (h + pad[0] + pad[2] - k) // stride + 1
    wo = (wd + pad[1] + pad[3] - k) // stride + 1
    y = np.zeros((o, ho, wo))
    og = o // group
    for g in range(group):
        xs = xp[g * cg:(g + 1) * cg]
        for kh in range(k):
            for kw in range(k):
                patch = xs[:, kh:kh + stride * ho:stride, kw:kw + stride * wo:stride]   # [cg, ho, wo]
                y[g * og:(g + 1) * og] += np.einsum("oc,chw->ohw", w[g * og:(g + 1) * og, :, kh, kw], patch)
    return y


def run(model, x):
    """x: uint8 input [C,H,W]. Returns {tensor name: array} and the layer-end tensor names in order."""
    g = model.graph
    T = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
    T[g.input[0].name] = x.astype(np.float64)
    ends, signed = [], {}
    for n in g.node:
        a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
        ins = [T.get(i) for i in n.input]
        op = n.op_type
        if op == "Constant":
            y = numpy_helper.to_array(a["value"]).astype(np.float64)
        elif op == "Conv":
            y = conv(ins[0], ins[1], a.get("strides", [1, 1])[0], a.get("pads", [0, 0, 0, 0]), a.get("group", 1))
            if len(ins) > 2 and ins[2] is not None:
                y += ins[2].reshape(-1, 1, 1)
        elif op == "Mul":
            y = ins[0] * ins[1]
        elif op == "Div":
            y = ins[0] / ins[1]
        elif op == "Floor":
            y = np.floor(ins[0])
        elif op == "Clip":
            y = np.clip(ins[0], a.get("min", ins[1] if len(ins) > 1 else None), a.get("max", ins[2] if len(ins) > 2 else None))
            signed[n.output[0]] = a.get("min", 0) < 0
            ends.append(n.output[0])
        elif op == "Add" and "out_mul" in a:   # Quantlab residual add
            # Quantlab residual add (matches DORY's golden outputs): rescale (and clip) the inputs flagged rq,
            # requantize; the result is signed when an input is signed
            lo = -128 if signed.get(n.input[0]) or signed.get(n.input[1]) else 0
            def rq(v, i):   # each rescaled input is clipped to 8 bits
                return np.clip(np.floor((v * a[f"in{i}_mul"] + a[f"in{i}_add"]) / 2.0 ** a[f"in{i}_shift"]), lo, lo + 255) if a[f"in{i}_rq"] else v
            y = np.clip(np.floor(((rq(ins[0], 1) + rq(ins[1], 2)) * a["out_mul"] + a["out_add"]) / 2.0 ** a["out_shift"]),
                        lo, lo + a["out_n_levels"] - 1)
            signed[n.output[0]] = lo < 0
            ends.append(n.output[0])
        elif op == "Add":
            y = ins[0] + ins[1]
        elif op == "GlobalAveragePool":
            y = ins[0].mean(axis=(1, 2), keepdims=True)
        elif op in ("Shape", "Gather", "Unsqueeze", "Concat"):
            y = None
        elif op == "Reshape":
            y = ins[0].reshape(-1)
        elif op == "Gemm":
            y = ins[1] @ ins[0] + (ins[2] if len(ins) > 2 else 0)
            ends.append(n.output[0])
        else:
            raise ValueError(op)
        T[n.output[0]] = y
    return T, ends


def read_golden(path):
    vals = [l.strip().rstrip(",") for l in open(path) if l.strip() and not l.startswith("#")]
    return np.array([float(v) for v in vals if v])


if __name__ == "__main__":
    import os
    model = onnx.load(sys.argv[1])
    d = sys.argv[2]
    shp = [x.dim_value for x in model.graph.input[0].type.tensor_type.shape.dim][1:]
    x = read_golden(os.path.join(d, "input.txt")).reshape(shp[1], shp[2], shp[0]).transpose(2, 0, 1)
    T, ends = run(model, x)
    print(len(ends), "layer outputs")
    bad = 0
    for i, name in enumerate(ends):
        f = os.path.join(d, f"out_layer{i}.txt")
        if not os.path.exists(f):
            continue
        gold = read_golden(f)
        y = T[name]
        y = y.transpose(1, 2, 0).ravel() if y.ndim == 3 else y.ravel()
        if y.size != gold.size or not np.array_equal(y, gold):
            bad += 1
            print("layer", i, name, "MISMATCH", y.size, gold.size, np.abs(y - gold).max() if y.size == gold.size else "")
    print("mismatching layers:", bad)
