"""Write data.h for the TFLite Micro app: the MLPerf Tiny int8 ResNet-8 model and the first N
CIFAR-10 test images (HWC, raw 0..255, the model's input is pixel - 128).
    python3 gen_data.py MODEL.tflite CIFAR_DIR [N]"""
import pickle, sys, os
import numpy as np
tfl, cifar = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
d = pickle.load(open(os.path.join(cifar, "cifar-10-batches-py", "test_batch"), "rb"), encoding="bytes")
x = d[b"data"][:N].reshape(N, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
y = np.array(d[b"labels"][:N])
m = open(tfl, "rb").read()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.h"), "w") as f:
    f.write(f"/* MLPerf Tiny ResNet-8 int8 ({os.path.basename(tfl)}) and the first {N} CIFAR-10 test images */\n")
    f.write(f"#define N_IMG {N}\nstatic const unsigned char LABELS[N_IMG] = {{{', '.join(map(str, y))}}};\n")
    f.write(f"static const unsigned char MODEL[{len(m)}] __attribute__((aligned(16))) = {{{','.join(map(str, m))}}};\n")
    f.write(f"static const unsigned char IMAGES[N_IMG][3072] = {{{','.join('{' + ','.join(map(str, im.ravel())) + '}' for im in x)}}};\n")
print("labels", y)
