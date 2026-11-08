"""Quantlab-style integer ONNX -> the int8 GAP8 engine's net header, weights file, input and reference.

    python gen8.py MODEL_INT.onnx INPUT.txt OUT_DIR NAME

Writes OUT_DIR/net8_NAME.h, NAME.w8 (weights for the flash), NAME.u8 (Winograd weights), NAME.in8 (HWC input bytes) and NAME.ref8
(qref.py's int32 logits). Layers, in DORY's order:
  CONV: y = clip((acc * k + l) >> d) to 0..255 or -128..127; dense weights [co][kh][kw][ci] padded
        so each output channel's row is a multiple of 4 bytes; depthwise weights [kh][kw][c].
        3x3 stride-1 dense convs also get Winograd F(2,3) weights U' = (2G) g (2G)^T = 4U, int16,
        [co][16][ci] (ci padded to even), so the engine can use either.
  ADD:  y = clip((rq1(a) + rq2(b)) * om + oa) >> os), rq(x) = clip((x * m + a) >> s) when flagged.
  GAP:  y = clip((mean * m + a) >> d) with the mean taken as sum * m / n in 64 bits.
  FC:   int32 = W x + b.
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

import qref

OP = {"CONV": 0, "ADD": 1, "GAP": 2, "FC": 3}
G2I = np.array([[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]])


def layers(model):
    """Walk the graph and group nodes into DORY layers."""
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consts = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node if n.op_type == "Constant"}
    prod = {o: n for n in g.node for o in n.output}
    shape = {g.input[0].name: [d.dim_value for d in g.input[0].type.tensor_type.shape.dim][1:]}
    signed = {g.input[0].name: False}
    out = []
    pending = {}
    alias = {}
    def src(t):
        while t in alias:
            t = alias[t]
        return t
    for n in g.node:
        a = {at.name: onnx.helper.get_attribute_value(at) for at in n.attribute}
        op = n.op_type
        if op == "Conv":
            w = init[n.input[1]].astype(np.int64)
            c, h, wd = shape[src(n.input[0])]
            k, s, p = a["kernel_shape"][0], a["strides"][0], a.get("pads", [0, 0, 0, 0])
            ho, wo = (h + p[0] + p[2] - k) // s + 1, (wd + p[1] + p[3] - k) // s + 1
            pending[n.output[0]] = dict(type="CONV", inp=src(n.input[0]), w=w, group=a.get("group", 1), k=k, s=s, pads=p,
                                        cin=c, cout=w.shape[0], shape=[w.shape[0], ho, wo])
        elif op == "GlobalAveragePool":
            c, h, wd = shape[src(n.input[0])]
            pending[n.output[0]] = dict(type="GAP", inp=src(n.input[0]), cin=c, n=h * wd, shape=[c, 1, 1])
        elif op in ("Mul", "Add", "Div", "Floor") and n.input[0] in pending:
            L = pending.pop(n.input[0])
            other = n.input[1] if len(n.input) > 1 else None
            v = init.get(other, consts.get(other))
            if op == "Mul":
                L["kmul"] = np.broadcast_to(v.ravel(), (L["shape"][0],)).astype(np.int64)
            elif op == "Add":
                L["ladd"] = np.broadcast_to(v.ravel(), (L["shape"][0],)).astype(np.int64)
            elif op == "Div":
                L["d"] = int(np.log2(float(v)))
                assert 2 ** L["d"] == float(v)
            pending[n.output[0]] = L
        elif op == "Clip" and n.input[0] in pending:
            L = pending.pop(n.input[0])
            L["lo"], L["hi"] = int(a["min"]), int(a["max"])
            L["out"] = n.output[0]
            shape[n.output[0]] = L["shape"]
            signed[n.output[0]] = L["lo"] < 0
            L["in_signed"] = signed[L["inp"]]
            out.append(L)
        elif op == "Add" and "out_mul" in a:
            i1, i2 = src(n.input[0]), src(n.input[1])
            lo = -128 if signed[i1] or signed[i2] else 0
            L = dict(type="ADD", inp=i1, inp2=i2, shape=shape[i1], lo=lo, hi=lo + 255, out=n.output[0],
                     rq=[(a[f"in{i}_mul"], a[f"in{i}_add"], a[f"in{i}_shift"], a[f"in{i}_rq"]) for i in (1, 2)],
                     om=a["out_mul"], oa=a["out_add"], os=a["out_shift"], in_signed=signed[i1], in2_signed=signed[i2])
            shape[n.output[0]] = L["shape"]
            signed[n.output[0]] = lo < 0
            out.append(L)
        elif op == "Reshape":
            alias[n.output[0]] = n.input[0]
        elif op == "Gemm":
            x = src(n.input[0])
            w = init[n.input[1]].astype(np.int64)
            b = init[n.input[2]].astype(np.int64) if len(n.input) > 2 else np.zeros(w.shape[0], np.int64)
            out.append(dict(type="FC", inp=x, w=w, b=b, nin=w.shape[1], nout=w.shape[0], shape=[w.shape[0], 1, 1],
                            out=n.output[0], in_signed=signed[x]))
            shape[n.output[0]] = [w.shape[0], 1, 1]
        elif op in ("Constant", "Shape", "Gather", "Unsqueeze", "Concat"):
            pass
        else:
            raise ValueError(f"unexpected {op}")
    return out, shape


def assign(lys, inp, out_name):
    """HyperRAM byte offsets: largest first, lowest address free for the whole lifetime."""
    sizes, first, last = {inp: int(np.prod(lys[0]["_inshape"]))}, {inp: -1}, {}
    for i, L in enumerate(lys):
        sizes[L["out"]] = int(np.prod(L["shape"])) * (4 if L["type"] == "FC" else 1)
        first[L["out"]] = i
        for t in (L["inp"], L.get("inp2")):
            if t:
                last[t] = i
    last[inp] = len(lys)
    last[out_name] = len(lys)
    for t in first:
        last.setdefault(t, first[t])
    base = {}
    for t in sorted(first, key=lambda t: -sizes[t]):
        need = (sizes[t] + 63) // 64 * 64
        busy = sorted((base[u], base[u] + (sizes[u] + 63) // 64 * 64) for u in base if first[u] <= last[t] and first[t] <= last[u])
        off = 0
        for lo, hi in busy:
            if off + need <= lo:
                break
            off = max(off, hi)
        base[t] = off
    return base, max(base[t] + sizes[t] for t in base)


def main():
    onnx_path, input_txt, out_dir, name = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(onnx_path)
    lys, shape = layers(model)
    inp = model.graph.input[0].name
    c, h, w = shape[inp]
    lys[0]["_inshape"] = [c, h, w]
    x = qref.read_golden(input_txt).reshape(h, w, c).transpose(2, 0, 1)
    T, ends = qref.run(model, x)
    assert len(ends) == len(lys), (len(ends), len(lys))
    for L, e in zip(lys, ends):
        assert L["out"] == e
    base, ram = assign(lys, inp, lys[-1]["out"])

    blob, ublob = bytearray(), bytearray()     # direct weights; Winograd weights (own file, NAME.u8)
    def put(arr, dtype, b=blob):
        off = len(b)
        b.extend(np.asarray(arr, dtype).tobytes())
        while len(b) % 4:
            b.append(0)
        return off
    rows = []
    for L in lys:
        r = dict(type=OP[L["type"]], inp=base[L["inp"]], inp2=base.get(L.get("inp2"), 0), out=base[L["out"]],
                 c=L["shape"][0], h=L["shape"][1], w=L["shape"][2], lo=L.get("lo", 0), hi=L.get("hi", 0),
                 in_signed=int(L["in_signed"]), woff=-1, uoff=-1, koff=-1, d=0, k=0, s=1, pads=[0, 0, 0, 0], cin=0,
                 ih=0, iw=0, group=1, kpad=0, p=[0] * 9)
        if L["type"] in ("CONV", "GAP", "ADD"):
            ci, ih, iw = shape[L["inp"]]
            r.update(cin=ci, ih=ih, iw=iw)
        if L["type"] == "CONV":
            W = L["w"]
            r.update(k=L["k"], s=L["s"], pads=L["pads"], group=L["group"], d=L["d"])
            if L["group"] == 1:
                Wr = W.transpose(0, 2, 3, 1).reshape(W.shape[0], -1)          # [co][kh][kw][ci]
                kpad = (Wr.shape[1] + 3) // 4 * 4
                r["kpad"] = kpad
                r["woff"] = put(np.pad(Wr, ((0, 0), (0, kpad - Wr.shape[1]))), np.int8)
                if (L["k"], L["s"], L["pads"]) == (3, 1, [1, 1, 1, 1]):
                    U = np.einsum("ri,ocij,sj->orsc", G2I, W, G2I).reshape(W.shape[0], 16, -1)   # [co][16][ci]
                    assert np.abs(U).max() < 32768
                    cpad = (U.shape[2] + 1) // 2 * 2
                    r["uoff"] = put(np.pad(U, ((0, 0), (0, 0), (0, cpad - U.shape[2]))), np.int16, ublob)
            else:
                r["woff"] = put(W.reshape(W.shape[0], -1).T, np.int8)      # [kh][kw][c]
            r["koff"] = put(np.stack([L["kmul"], L["ladd"]], 1), np.int32)   # k, l per channel
        elif L["type"] == "GAP":
            r.update(d=L["d"], p=[int(L["kmul"][0]), int(L["ladd"][0]), L["n"]] + [0] * 6)
        elif L["type"] == "ADD":
            (m1, a1, s1, q1), (m2, a2, s2, q2) = L["rq"]
            r.update(in_signed=int(L["in_signed"]) | (int(L["in2_signed"]) << 1), p=[m1, a1, s1 | (q1 << 8), m2, a2, s2 | (q2 << 8), L["om"], L["oa"], L["os"]])
        elif L["type"] == "FC":
            r.update(cin=L["nin"], woff=put(L["w"], np.int8), koff=put(L["b"], np.int32))
        rows.append(r)

    open(os.path.join(out_dir, f"{name}.w8"), "wb").write(bytes(blob))
    open(os.path.join(out_dir, f"{name}.u8"), "wb").write(bytes(ublob) or bytes(4))
    np.asarray(x.transpose(1, 2, 0).ravel(), np.int64).astype(np.uint8).tofile(os.path.join(out_dir, f"{name}.in8"))
    np.asarray(T[lys[-1]["out"]], np.int64).astype(np.int32).tofile(os.path.join(out_dir, f"{name}.ref8"))
    # every layer's output, for debugging the engine on the laptop
    with open(os.path.join(out_dir, f"{name}.layers8"), "wb") as f:
        for L, e in zip(lys, ends):
            v = T[e]
            v = v.transpose(1, 2, 0).ravel() if v.ndim == 3 else v.ravel()
            np.asarray(v, np.int64).astype(np.int32 if L["type"] == "FC" else np.int8 if L["lo"] < 0 else np.uint8).tofile(f)
    with open(os.path.join(out_dir, f"net8_{name}.h"), "w") as f:
        f.write(f"/* generated by gen8.py from {os.path.basename(onnx_path)}: {len(rows)} layers, {ram / 1e6:.2f} MB of HyperRAM, "
                f"{len(blob) / 1e6:.2f} MB of weights */\n")
        f.write("enum { L_CONV, L_ADD, L_GAP, L_FC };\n")
        f.write("typedef struct { int type; uint32_t in, in2, out; int c, h, w, cin, ih, iw, k, s, pads[4], group, kpad, d, lo, hi, "
                "in_signed; int woff, uoff, koff; int p[9]; } Layer;\n")
        f.write(f"#define N_LAYERS {len(rows)}\n#define RAM_USED {ram}u\n#define IN_OFF {base[inp]}u\n#define IN_BYTES {c * h * w}\n")
        f.write(f"#define OUT_N {lys[-1]['shape'][0]}\n")
        f.write("static const Layer LAYERS[N_LAYERS] = {\n")
        for r in rows:
            f.write(f"    {{{r['type']}, {r['inp']}u, {r['inp2']}u, {r['out']}u, {r['c']}, {r['h']}, {r['w']}, {r['cin']}, {r['ih']}, "
                    f"{r['iw']}, {r['k']}, {r['s']}, {{{', '.join(map(str, r['pads']))}}}, {r['group']}, {r['kpad']}, {r['d']}, "
                    f"{r['lo']}, {r['hi']}, {r['in_signed']}, {r['woff']}, {r['uoff']}, {r['koff']}, {{{', '.join(map(str, r['p']))}}}}},\n")
        f.write("};\n")
    print(f"{name}: {len(rows)} layers, {ram / 1e6:.2f} MB HyperRAM, {len(blob) / 1e6:.2f} MB weights,"
          f" {sum(r['uoff'] >= 0 for r in rows)} Winograd-capable convs")


if __name__ == "__main__":
    main()
