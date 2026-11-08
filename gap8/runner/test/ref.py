"""numpy reference for model_runner plans, used to check the GAP8 engine on made-up graphs.
Usage: python3 ref.py NAME  -> NAME.plan/.bin/.input/.ref (random weights and input)"""
import sys
import numpy as np

rng = np.random.default_rng(1)
name = sys.argv[1]

def conv(x, w, b, s, p, groups=1):
    c, h, wd = x.shape
    co, cig, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    ho, wo = (h + 2 * p - k) // s + 1, (wd + 2 * p - k) // s + 1
    y = np.zeros((co, ho, wo), np.float64)
    for o in range(co):
        g = o // (co // groups)
        for i in range(cig):
            for kh in range(k):
                for kw in range(k):
                    y[o] += w[o, i, kh, kw] * xp[g * cig + i, kh:kh + s * ho:s, kw:kw + s * wo:s]
        if b is not None:
            y[o] += b[o]
    return y

# a graph using every op: id -> array
lines, W, T = [], [], {}
def store(a):
    off = sum(x.size for x in W); W.append(a.astype(np.float32).ravel()); return off
def tensor(i, a):
    T[i] = a; lines.append(f"TENSOR {i} {a.shape[0]} {a.shape[1]} {a.shape[2]}")
def cv(out, inp, cout, k, s, p, act=None, dw=False):
    x = T[inp]; cin = x.shape[0]
    w = rng.standard_normal((cout, 1 if dw else cin, k, k)) / np.sqrt(k * k * (1 if dw else cin))
    b = rng.standard_normal(cout) * 0.1
    y = conv(x, w, b, s, p, groups=cin if dw else 1)
    tensor(out, y)
    wo, bo = store(w), store(b)
    lines.append(f"DWCONV {out} {inp} {cin} {k} {s} {p} {wo} {bo}" if dw else f"CONV {out} {inp} {cin} {cout} {k} {s} {p} 4 {wo} {bo}")
    if act == "relu":
        T[out] = np.maximum(y, 0); lines.append(f"RELU {out} {out}")
    elif act == "relu6":
        T[out] = np.clip(y, 0, 6); lines.append(f"CLIP {out} {out} 0.0 6.0")
    elif act == "silu":
        T[out] = y / (1 + np.exp(-y)); lines.append(f"SILU {out} {out}")

H, Wd = (19, 23) if name == "t1" else (37, 30)
x = rng.standard_normal((3, H, Wd))
tensor(0, x); lines.append("INPUT 0")
cv(1, 0, 16, 3, 1, 1, "relu")                # Winograd (or direct by the cost rule)
cv(2, 1, 24, 3, 2, 1, "silu")                # 3x3 stride 2
cv(3, 2, 24, 3, 1, 1, "relu6", dw=True)      # depthwise s1
cv(4, 3, 24, 3, 2, 1, None, dw=True)         # depthwise s2
cv(5, 4, 32, 1, 1, 0, None)                  # 1x1
tensor(6, T[5][:16]); tensor(7, T[5][16:]); lines.append("SPLIT 6 7 5")
cv(8, 7, 16, 3, 1, 1, "silu")
tensor(9, T[8] + T[6]); lines.append("ADD 9 8 6")
tensor(10, np.maximum(T[9], 0)); lines.append("RELU 10 9")
mp = np.pad(T[10], ((0, 0), (2, 2), (2, 2)), constant_values=-np.inf)
h10, w10 = T[10].shape[1:]
tensor(11, np.max([mp[:, i:i + h10, j:j + w10] for i in range(5) for j in range(5)], axis=0)); lines.append("MAXPOOL 11 10 5 1 2")
tensor(12, np.concatenate([T[11], T[6], T[7]])); lines.append("CONCAT 12 3 11 6 7")
tensor(13, T[12].repeat(2, 1).repeat(2, 2)); lines.append("UPSAMPLE 13 12")
cv(14, 13, 40, 3, 1, 1, "relu")
cv(15, 14, 40, 1, 2, 0, None)                # 1x1 stride 2
tensor(16, np.clip(T[15], -1.0, 2.5)); lines.append("CLIP 16 15 -1.0 2.5")
tensor(17, T[16].mean(axis=(1, 2)).reshape(-1, 1, 1)); lines.append("GAP 17 16")
gw = rng.standard_normal((13, 40)); gb = rng.standard_normal(13)
tensor(18, (gw @ T[17].ravel() + gb).reshape(-1, 1, 1)); lines.append(f"GEMM 18 17 40 13 {store(gw)} {store(gb)}")
lines += ["OUTPUT 18", "OUTPUT 14"]
open(f"{name}.plan", "w").write("\n".join(lines) + "\n")
np.concatenate(W).tofile(f"{name}.bin")
x.astype(np.float32).tofile(f"{name}.input")
np.concatenate([T[18].ravel(), T[14].ravel()]).astype(np.float32).tofile(f"{name}.ref")
print(name, "ok", T[14].shape)
