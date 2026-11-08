"""Turn a model_runner plan into the GAP8 engine's net header.

Usage: python3 gen.py MODEL.plan net_MODEL.h

The weights file (.bin) is used as is: the header keeps the plan's float offsets into it.
MODEL.u gets Winograd F(4,3)-transformed weights for as many 3x3 stride-1 convs as the flash
has room for (the chip transforms the others on every run).
An activation that follows the conv writing the same tensor is folded into that conv, as
model_runner does. HyperRAM offsets are shared between tensors whose lifetimes don't overlap: largest first,
each at the lowest address that is free for its whole lifetime.
"""
import sys

TYPES = {"CONV": "OP_CONV", "DWCONV": "OP_DWCONV", "RELU": "OP_RELU", "CLIP": "OP_CLIP", "SILU": "OP_SILU",
         "ADD": "OP_ADD", "MAXPOOL": "OP_MAXPOOL", "GAP": "OP_GAP", "GEMM": "OP_GEMM", "CONCAT": "OP_CONCAT",
         "UPSAMPLE": "OP_UPSAMPLE"}
RAM_BYTES = 8 << 20
FLASH_BUDGET = 60 << 20  # of the GAPuino's 64 MB HyperFlash, for all readfs files
G4 = [[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6], [-1 / 6, 1 / 6, -1 / 6], [1 / 24, 1 / 12, 1 / 6], [1 / 24, -1 / 12, 1 / 6],
      [0, 0, 1]]


def pretransform(ops, plan):
    """Append U = G g G^T ([cout][cin][36]) to MODEL.u for 3x3 stride-1 convs while the flash has room."""
    import os
    import numpy as np
    base = plan[:-len(".plan")]
    used = sum(os.path.getsize(base + e) for e in (".bin", ".input", ".ref"))
    G = np.array(G4)
    blocks, off = [], 0
    for o in ops:
        o["uoff"] = -1
        if o["type"] != "CONV" or (o["k"], o["s"], o["p"]) != (3, 1, 1):
            continue
        n = o["cout"] * o["cin"] * 36
        if used + (off + n) * 4 > FLASH_BUDGET:
            continue
        w = np.fromfile(base + ".bin", np.float32, o["cout"] * o["cin"] * 9, offset=o["woff"] * 4)
        u = np.einsum("ri,ocij,sj->ocrs", G, w.reshape(o["cout"], o["cin"], 3, 3).astype(np.float64), G)
        blocks.append(u.astype(np.float32).ravel())
        o["uoff"] = off
        off += n
    data = np.concatenate(blocks) if blocks else np.zeros(1, np.float32)
    data.tofile(base + ".u")
    return off * 4


def parse(path):
    tens, ops, alias, outs, inp = {}, [], {}, [], None
    for line in open(path):
        f = line.split()
        if not f:
            continue
        w, a = f[0], f[1:]
        if w == "TENSOR":
            tens[int(a[0])] = tuple(int(x) for x in a[1:4])
        elif w == "INPUT":
            inp = int(a[0])
        elif w == "OUTPUT":
            outs.append(int(a[0]))
        elif w == "SPLIT":
            x, y, src = (int(v) for v in a)
            alias[x] = (src, 0)
            alias[y] = (src, tens[x][0] * tens[x][1] * tens[x][2])
        else:
            o = dict(type=w, act=0, lo=0.0, hi=0.0, in2=0, ins=[], cin=0, cout=0, k=0, s=0, p=0, nin=0, nout=0,
                     woff=-1, boff=-1, uoff=-1)
            if w == "CONV":
                o.update(zip(["out", "in", "cin", "cout", "k", "s", "p", "m", "woff", "boff"], map(int, a)))
            elif w == "DWCONV":
                o.update(zip(["out", "in", "cin", "k", "s", "p", "woff", "boff"], map(int, a)))
                o["cout"] = o["cin"]
            elif w == "CLIP":
                o.update(out=int(a[0]), **{"in": int(a[1])}, lo=float(a[2]), hi=float(a[3]))
            elif w in ("RELU", "SILU", "UPSAMPLE", "GAP"):
                o.update(out=int(a[0]), **{"in": int(a[1])})
            elif w == "ADD":
                o.update(out=int(a[0]), **{"in": int(a[1])}, in2=int(a[2]))
            elif w == "MAXPOOL":
                o.update(zip(["out", "in", "k", "s", "p"], map(int, a)))
            elif w == "GEMM":
                o.update(zip(["out", "in", "nin", "nout", "woff", "boff"], map(int, a)))
            elif w == "CONCAT":
                n = int(a[1])
                o.update(out=int(a[0]), ins=[int(x) for x in a[2:2 + n]])
                o["in"] = o["ins"][0]
            else:
                raise ValueError(w)
            prev = ops[-1] if ops else None
            if (prev and prev["type"] in ("CONV", "DWCONV") and not prev["act"] and o["in"] == o["out"] == prev["out"]
                    and w in ("RELU", "CLIP", "SILU")):
                prev["act"] = 2 if w == "SILU" else 1
                prev["lo"] = 0.0 if w == "RELU" else o["lo"]
                prev["hi"] = float("inf") if w == "RELU" else o["hi"]
                continue
            ops.append(o)
    return tens, ops, alias, outs, inp


def assign(tens, ops, alias, outs, inp):
    root = lambda t: root(alias[t][0]) if t in alias else t
    size = lambda t: tens[t][0] * tens[t][1] * tens[t][2] * 4
    last = {}
    for i, o in enumerate(ops):
        reads = o["ins"] or [o["in"]] + ([o["in2"]] if o["type"] == "ADD" else [])
        for t in reads:
            last[root(t)] = i
        r = root(o["out"])
        last[r] = max(last.get(r, -1), i)
    forever = len(ops) + 1
    last[root(inp)] = forever
    for t in outs:
        last[root(t)] = forever
    first = {root(inp): -1}
    for i, o in enumerate(ops):
        first.setdefault(root(o["out"]), i)
    # Largest tensors first, each at the lowest address free for its whole lifetime
    placed, base = [], {}
    for t in sorted(first, key=lambda t: -size(t)):
        need = (size(t) + 63) // 64 * 64
        busy = sorted((base[u], base[u] + (size(u) + 63) // 64 * 64) for u in placed
                      if first[u] <= last[t] and first[t] <= last[u])
        off = 0
        for a, b in busy:
            if off + need <= a:
                break
            off = max(off, b)
        base[t] = off
        placed.append(t)
    off = max(base[t] + size(t) for t in placed)
    if off > RAM_BYTES:
        raise SystemExit(f"activations need {off / 1e6:.1f} MB, more than the 8 MB HyperRAM")
    offs = {}
    for t in tens:
        r = root(t)
        if r in base:
            offs[t] = base[r] + (alias[t][1] * 4 if t in alias else 0)
    return offs, off


def main():
    plan, out = sys.argv[1], sys.argv[2]
    tens, ops, alias, outs, inp = parse(plan)
    offs, total = assign(tens, ops, alias, outs, inp)
    ubytes = pretransform(ops, plan)
    n = max(tens) + 1
    flt = lambda v: "INFINITY" if v == float("inf") else repr(float(v)) + "f"
    with open(out, "w") as f:
        f.write(f"/* generated by gen.py from {plan.split('/')[-1]}: {len(ops)} ops, {total / 1e6:.2f} MB of HyperRAM */\n")
        f.write("enum { OP_CONV, OP_DWCONV, OP_RELU, OP_CLIP, OP_SILU, OP_ADD, OP_MAXPOOL, OP_GAP, OP_GEMM, OP_CONCAT, "
                "OP_UPSAMPLE };\nenum { ACT_NONE, ACT_CLAMP, ACT_SILU };\n")
        f.write("typedef struct { int c, h, w; uint32_t off; } Ten;\n")
        f.write("typedef struct { short type, out, in, in2, cin, cout, k, s, p, act; float lo, hi; "
                "short n_ins, ins[8]; int nin, nout, woff, boff, uoff; } Op;\n")
        f.write(f"#define N_TENS {n}\n#define N_OPS {len(ops)}\n#define IN_ID {inp}\n#define RAM_USED {total}u\n")
        f.write(f"#define N_OUTS {len(outs)}\nstatic const int OUTS[N_OUTS] = {{{', '.join(map(str, outs))}}};\n")
        f.write("static const Ten TENS[N_TENS] = {\n")
        for t in range(n):
            c, h, w = tens.get(t, (0, 0, 0))
            f.write(f"    {{{c}, {h}, {w}, {offs.get(t, 0)}u}},\n")
        f.write("};\nstatic const Op OPS[N_OPS] = {\n")
        for o in ops:
            ins = o["ins"] + [0] * (8 - len(o["ins"]))
            f.write(f"    {{{TYPES[o['type']]}, {o['out']}, {o['in']}, {o['in2']}, {o['cin']}, {o['cout']}, {o['k']}, "
                    f"{o['s']}, {o['p']}, {o['act']}, {flt(o['lo'])}, {flt(o['hi'])}, {len(o['ins'])}, "
                    f"{{{', '.join(map(str, ins))}}}, {o['nin']}, {o['nout']}, {o['woff']}, {o['boff']}, {o['uoff']}}},\n")
        f.write("};\n")
    npre = sum(o["uoff"] >= 0 for o in ops)
    print(f"{out}: {len(ops)} ops, {total / 1e6:.2f} MB HyperRAM, {npre} convs pretransformed ({ubytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
