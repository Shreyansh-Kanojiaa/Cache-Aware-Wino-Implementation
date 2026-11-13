"""On the Pi: export trained models to ONNX for the engine comparison, and check that our engine
predicts the same classes as PyTorch on real validation images.

    python export_pi.py NAME [NAME ...]     (NAME.pt and NAME_val.npz next to this file)
Writes ~/wino-own/woof_NAME.onnx (input 1x3x192x192) and prints the engine-vs-PyTorch agreement.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from winonet import build

OWN = os.path.expanduser("~/wino-own")
sys.path.insert(0, OWN)
from benchmarks.whole_model import RUNNER, export_plan   # noqa: E402

for name in sys.argv[1:]:
    here = os.path.dirname(os.path.abspath(__file__))
    ck = torch.load(os.path.join(here, f"{name}.pt"), map_location="cpu")
    model = build("resnet18" if name == "resnet18" else "winonet", ck["spec"])
    if name != "resnet18":
        model.fuse()
    model.load_state_dict(ck["state"])
    model.eval()
    onnx_path = os.path.join(OWN, f"woof_{name}.onnx")   # woof_: never clash with the benchmark models
    kw = {"dynamo": False} if int(torch.__version__.split(".")[0]) >= 2 else {}   # PyTorch 1.x has only this exporter
    torch.onnx.export(model, torch.zeros(1, 3, 192, 192), onnx_path, input_names=["input"], output_names=["logits"],
                      opset_version=13, **kw)
    plan, weights, _, _ = export_plan(f"woof_{name}")
    val = np.load(os.path.join(here, f"{name}_val.npz"))
    agree = correct = 0
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for x, y in zip(val["x"], val["y"]):
            inp, out = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
            x.astype(np.float32).tofile(inp)
            subprocess.run([RUNNER, "--plan", plan, "--weights", weights, "--input", inp, "--output", out,
                            "--csv", os.path.join(tmp, "r.csv"), "--variant", "v6_freq", "--T", "4",
                            "--warmup", "0", "--runs", "1"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ours = np.fromfile(out, dtype=np.float32)
            ref = model(torch.from_numpy(x[None])).numpy().ravel()
            agree += int(ours.argmax() == ref.argmax())
            correct += int(ours.argmax() == y)
    print(f"{name}: trained val {ck['val_acc'] * 100:.2f}%, {ck['gmac']:.2f} GMAC; on {len(val['y'])} val images our engine "
          f"agrees with PyTorch on {agree}, correct on {correct}", flush=True)
