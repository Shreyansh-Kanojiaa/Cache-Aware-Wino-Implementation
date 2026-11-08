"""Train WinoNet or the ResNet-18 baseline on Imagewoof at 192 px, with one recipe for both.

    python train.py DATA_DIR NAME [EPOCHS] [SPEC_JSON]
      DATA_DIR  holds imagewoof2-320/{train,val}
      NAME      resnet18, or a WinoNet name whose spec is SPEC_JSON (e.g. '{"stem":32,"stages":[[48,2],...]}')
Writes NAME.pt (weights, fused for WinoNet, plus the spec), NAME.log, and NAME_val.npy (a few val
images and their labels, to check the Pi engine's predictions).

Images are decoded once into a uint8 cache (256x256 center crops). Augmentation runs on the GPU:
a random resized crop (scale 0.25-1, aspect 3/4-4/3) with flips, done as one affine_grid call.
Validation: center 224 of the 256 crop, resized to 192 (the usual 0.875 crop ratio).
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from winonet import build, flops

MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1) * 255


def load(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 256) // 2, (h - 256) // 2
    return np.asarray(im.crop((l, t, l + 256, t + 256)), dtype=np.uint8)


def cache(data_dir, split):
    path = os.path.join(data_dir, f"woof_{split}_256.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["x"], d["y"]
    root = os.path.join(data_dir, "imagewoof2-320", split)
    classes = sorted(os.listdir(root))
    files = [(os.path.join(root, c, f), i) for i, c in enumerate(classes) for f in sorted(os.listdir(os.path.join(root, c)))]
    with ProcessPoolExecutor(8) as pool:
        x = np.stack(list(pool.map(load, [f for f, _ in files], chunksize=64)))
    y = np.array([i for _, i in files], dtype=np.int64)
    np.savez(path, x=x, y=y)
    return x, y


def augment(x):
    """x: uint8 NHWC on GPU -> normalized NCHW 192x192 random resized crops."""
    n = x.shape[0]
    x = x.permute(0, 3, 1, 2).float()
    area = torch.empty(n, device=x.device).uniform_(0.25, 1.0)
    logr = torch.empty(n, device=x.device).uniform_(math.log(3 / 4), math.log(4 / 3))
    r = logr.exp()
    sw, sh = (area * r).sqrt().clamp(max=1), (area / r).sqrt().clamp(max=1)
    tx, ty = (torch.rand(n, device=x.device) * 2 - 1) * (1 - sw), (torch.rand(n, device=x.device) * 2 - 1) * (1 - sh)
    flip = torch.where(torch.rand(n, device=x.device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = sw * flip, tx, sh, ty
    grid = F.affine_grid(theta, (n, 3, 192, 192), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def val_view(x):
    x = x.permute(0, 3, 1, 2).float()[:, :, 16:240, 16:240]
    x = F.interpolate(x, size=(192, 192), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def evaluate(model, xv, yv, dev):
    model.eval()
    correct = 0
    for i in range(0, len(xv), 256):
        xb = val_view(torch.from_numpy(xv[i:i + 256]).to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correct += (model(xb).argmax(1).cpu() == torch.from_numpy(yv[i:i + 256])).sum().item()
    return correct / len(xv)


def main():
    data_dir, name = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    spec = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"
    xt, yt = cache(data_dir, "train")
    xv, yv = cache(data_dir, "val")
    model = build("resnet18" if name == "resnet18" else "winonet", spec)
    gmac = flops(model) / 1e9
    model = model.to(dev).to(memory_format=torch.channels_last)
    bs, steps = 64, len(xt) // 64
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    rest = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-5}, {"params": rest, "weight_decay": 0}],
                          lr=0.1, momentum=0.9, nesterov=True)
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (2 * steps)) * 0.5 * (1 + math.cos(math.pi * s / total)))
    xt_gpu = torch.from_numpy(xt).to(dev)          # 9025 x 256 x 256 x 3 uint8, 1.8 GB
    yt_gpu = torch.from_numpy(yt).to(dev)
    log = open(f"{name}.log", "w")
    print(f"{name}: {gmac:.2f} GMAC at 192, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params (train-time)", file=log, flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            xb = augment(xt_gpu[idx]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb), yt_gpu[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == epochs - 1:
            acc = evaluate(model, xv, yv, dev)
            print(f"epoch {ep + 1} loss {tot / steps:.3f} val {acc * 100:.2f}% ({time.time() - t0:.0f}s)", file=log, flush=True)
    if name != "resnet18":
        model.fuse()
    acc = evaluate(model, xv, yv, dev)
    print(f"final val {acc * 100:.2f}% ({'fused' if name != 'resnet18' else 'plain'})", file=log, flush=True)
    torch.save({"spec": spec, "state": model.float().cpu().state_dict(), "val_acc": acc, "gmac": gmac}, f"{name}.pt")
    # 64 val images (as the 192 crop, normalized) and labels, to check predictions on the Pi
    rng = np.random.RandomState(0)
    pick = rng.choice(len(xv), 64, replace=False)
    np.savez(f"{name}_val.npz", x=val_view(torch.from_numpy(xv[pick])).numpy(), y=yv[pick])


if __name__ == "__main__":
    main()
