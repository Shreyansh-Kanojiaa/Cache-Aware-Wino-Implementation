"""Train the MLPerf Tiny ResNet-8 on CIFAR-10 (laptop GPU) and save its weights.

Usage: python train_resnet8.py DATA_DIR OUT.pt [EPOCHS]
DATA_DIR holds cifar-10-batches-py (the official python version).
"""
import pickle, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class ResNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.stacks = nn.Sequential(Block(16, 16, 1), Block(16, 32, 2), Block(32, 64, 2))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stacks(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)


def load(data_dir):
    d = os.path.join(data_dir, "cifar-10-batches-py")
    def batch(name):
        with open(os.path.join(d, name), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        return b[b"data"].reshape(-1, 3, 32, 32), np.array(b[b"labels"])
    tr = [batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr, ytr = np.concatenate([t[0] for t in tr]), np.concatenate([t[1] for t in tr])
    xte, yte = batch("test_batch")
    norm = lambda x: (x.astype(np.float32) / 255.0 - MEAN) / STD
    return norm(xtr), ytr, norm(xte), yte


def main():
    data_dir, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr, xte, yte = load(data_dir)
    xtr, ytr = torch.tensor(xtr, device=dev), torch.tensor(ytr, device=dev)
    xte, yte = torch.tensor(xte, device=dev), torch.tensor(yte, device=dev)
    net = ResNet8().to(dev)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = epochs * (len(xtr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=steps)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(len(xtr) // 128):
            idx = perm[i * 128:(i + 1) * 128]
            x = xtr[idx]
            # random crop (pad 4) and horizontal flip
            x = F.pad(x, (4, 4, 4, 4), mode="reflect")
            dx, dy = np.random.randint(0, 9, 2)
            x = x[:, :, dy:dy + 32, dx:dx + 32]
            if np.random.rand() < 0.5:
                x = x.flip(3)
            loss = F.cross_entropy(net(x), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            acc = sum((net(xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum().item() for i in range(0, 10000, 1000)) / 100
        print(f"epoch {ep + 1} loss {loss.item():.3f} test {acc:.2f}%", flush=True)
    torch.save(net.state_dict(), out)
    np.save(out.replace(".pt", "_test_x.npy"), xte[:1000].cpu().numpy())
    np.save(out.replace(".pt", "_test_y.npy"), yte[:1000].cpu().numpy())


if __name__ == "__main__":
    main()
