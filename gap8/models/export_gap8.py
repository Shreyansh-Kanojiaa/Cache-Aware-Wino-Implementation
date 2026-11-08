"""Export models for the GAP8 engine (run on the Pi, from the harness-multicore checkout).

Usage: python3 export_gap8.py OUT_DIR MODEL...
MODEL: resnet8 (needs resnet8.pt and resnet8_test_x.npy next to this script), resnet18,
mobilenet_v2 (the Pi benchmark's ONNX files and inputs), yolov8n_320 (YOLOv8n at 320x320,
since its 640x640 activations don't fit the GAP8's 8 MB HyperRAM).

Writes MODEL.plan, MODEL.bin (weights), MODEL.input (N inputs) and MODEL.ref (ONNX Runtime's
outputs for each input, concatenated in OUTPUT order).
"""
import os, shutil, sys
import numpy as np
import torch
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
from benchmarks import whole_model as wm  # noqa: E402
from benchmarks.e2e_benchmark import _calibrate_batch_norm  # noqa: E402

N_RESNET8 = 8  # CIFAR-10 test images run on the board


def export_resnet8(path):
    sys.path.insert(0, HERE)
    from train_resnet8 import ResNet8
    net = ResNet8()
    net.load_state_dict(torch.load(os.path.join(HERE, "resnet8.pt"), map_location="cpu"))
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 3, 32, 32), path, opset_version=13, input_names=["input"],
                      output_names=["output"])
    x = np.load(os.path.join(HERE, "resnet8_test_x.npy"))[:N_RESNET8]
    return x.astype(np.float32)


def export_yolov8n_320(path):
    from ultralytics.nn.tasks import DetectionModel
    torch.manual_seed(0)
    model = DetectionModel("yolov8n.yaml", nc=80, verbose=False).eval()
    head = model.model[-1]
    head.forward = lambda x: [torch.cat((head.cv2[i](x[i]), head.cv3[i](x[i])), 1) for i in range(head.nl)]
    _calibrate_batch_norm(model, (3, 320, 320))
    torch.onnx.export(model, torch.randn(1, 3, 320, 320), path, opset_version=13, input_names=["input"],
                      output_names=["out0", "out1", "out2"])
    return np.random.default_rng(0).standard_normal((1, 3, 320, 320)).astype(np.float32)


def main():
    out_dir, models = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    for m in models:
        onnx_path = os.path.join(wm.ROOT, f"{m}.onnx")
        x = None
        if m == "resnet8":
            x = export_resnet8(onnx_path)
        elif m == "yolov8n_320":
            x = export_yolov8n_320(onnx_path)
        plan, weights, onnx_path, shape = wm.export_plan(m)
        if x is None:
            x = np.fromfile(os.path.join(wm.PLAN_DIR, f"{m}.input"), dtype=np.float32).reshape((1,) + tuple(shape))
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        refs = [np.concatenate([o.ravel() for o in sess.run(None, {name: xi[None]})]) for xi in x]
        shutil.copy(plan, os.path.join(out_dir, f"{m}.plan"))
        shutil.move(weights, os.path.join(out_dir, f"{m}.bin"))
        x.tofile(os.path.join(out_dir, f"{m}.input"))
        np.concatenate(refs).astype(np.float32).tofile(os.path.join(out_dir, f"{m}.ref"))
        print(m, "inputs", len(x), "outputs per input", refs[0].size, "weights MB",
              os.path.getsize(os.path.join(out_dir, f"{m}.bin")) / 1e6, flush=True)


if __name__ == "__main__":
    main()
