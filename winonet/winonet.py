"""WinoNet: an image classifier shaped for our Winograd engine on the Raspberry Pi 4.

At inference it is only 3x3 convolutions (+ bias + ReLU), stride-2 3x3 convolutions to
downsample, global average pooling and one small fully connected layer. During training each
3x3 conv has a parallel 1x1 branch and, where shapes allow, an identity branch (RepVGG,
Ding et al. 2021); fuse() folds them and the batch norms into a single 3x3 conv.

Image sizes 192 -> 96 -> 48 -> 24 -> 12 are all multiples of the engine's 4x4 Winograd output
tile, so no tile is half empty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k, s):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout))


def fold(conv_w, bn):
    """Weight and bias of conv followed by batch norm (eval mode)."""
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    return conv_w * t.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * t


class RepBlock(nn.Module):
    def __init__(self, cin, cout, s):
        super().__init__()
        self.cin, self.cout, self.s = cin, cout, s
        self.b3 = conv_bn(cin, cout, 3, s)
        self.b1 = conv_bn(cin, cout, 1, s)
        self.id = nn.BatchNorm2d(cout) if cin == cout and s == 1 else None
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return F.relu(self.fused(x))
        y = self.b3(x) + self.b1(x)
        if self.id is not None:
            y = y + self.id(x)
        return F.relu(y)

    @torch.no_grad()
    def fuse(self):
        w, b = fold(self.b3[0].weight, self.b3[1])
        w1, b1 = fold(self.b1[0].weight, self.b1[1])
        w = w + F.pad(w1, [1, 1, 1, 1])
        b = b + b1
        if self.id is not None:
            eye = torch.zeros(self.cout, self.cin, 3, 3, device=w.device)
            eye[torch.arange(self.cout), torch.arange(self.cin), 1, 1] = 1
            wi, bi = fold(eye, self.id)
            w, b = w + wi, b + bi
        self.fused = nn.Conv2d(self.cin, self.cout, 3, self.s, 1, bias=True).to(w.device)
        self.fused.weight.copy_(w)
        self.fused.bias.copy_(b)
        del self.b3, self.b1, self.id


class WinoNet(nn.Module):
    """stages: list of (channels, blocks); each stage after the first halves the image, with a
    stride-2 first block, or with pool=True a 2x2 max pool before it (faster on our engine:
    the conv then stays on the Winograd path)."""

    def __init__(self, stem, stages, classes=10, pool=False):
        super().__init__()
        layers, c = [RepBlock(3, stem, 2)], stem          # 192 -> 96
        for i, (ch, n) in enumerate(stages):
            for j in range(n):
                down = j == 0 and i > 0
                if down and pool:
                    layers.append(nn.MaxPool2d(2))
                layers.append(RepBlock(c, ch, 2 if down and not pool else 1))
                c = ch
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(c, classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.body(x), 1), 1))

    def fuse(self):
        for m in self.body:
            if isinstance(m, RepBlock):
                m.fuse()
        return self


# ---- baseline: ResNet-18 (torchvision layout, written out because the training venv has no torchvision)
class Basic(nn.Module):
    def __init__(self, cin, cout, s):
        super().__init__()
        self.c1, self.c2 = conv_bn(cin, cout, 3, s), conv_bn(cout, cout, 3, 1)
        self.down = conv_bn(cin, cout, 1, s) if s != 1 or cin != cout else None

    def forward(self, x):
        y = self.c2(F.relu(self.c1(x)))
        return F.relu(y + (x if self.down is None else self.down(x)))


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        blocks, c = [], 64
        for i, ch in enumerate((64, 128, 256, 512)):
            for j in range(2):
                blocks.append(Basic(c, ch, 2 if j == 0 and i > 0 else 1))
                c = ch
        self.body = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.body(self.stem(x)), 1), 1))


def build(name, spec=None):
    if name == "resnet18":
        return ResNet18()
    return WinoNet(spec["stem"], spec["stages"], pool=spec.get("pool", False))


def flops(model, size=192):
    """Multiply-adds of the convs and the fc, by hooks."""
    total = [0]
    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            total[0] += o.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hs:
        h.remove()
    return total[0]


if __name__ == "__main__":   # fused model must match the training-time model
    torch.manual_seed(0)
    m = WinoNet(32, [(32, 2), (64, 3), (128, 4), (256, 2)], pool=True)
    for mod in m.modules():          # non-trivial batch norm statistics
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-1, 1); mod.running_var.uniform_(0.5, 2)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.5, 0.5)
    m.eval()
    x = torch.randn(2, 3, 192, 192)
    a = m(x)
    b = m.fuse()(x)
    err = ((a - b).abs().max() / a.abs().max()).item()
    assert err < 1e-4, err
    print(f"fuse ok, rel err {err:.1e}; WinoNet {flops(m) / 1e9:.2f} GMAC, ResNet-18 {flops(ResNet18()) / 1e9:.2f} GMAC at 192")
