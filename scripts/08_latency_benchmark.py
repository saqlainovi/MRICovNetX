"""
=============================================================================
MRICovNetX - Experiment 8: Unified inference-latency benchmark
=============================================================================
Every architecture reported in the baseline-comparison table is timed with the
SAME protocol so that the latency column is directly comparable:

* input 1 x 3 x 224 x 224 (batch size 1), FP32, eval mode, torch.no_grad()
* 50 warm-up passes, then 5 repetitions x 200 timed passes
* torch.cuda.synchronize() before and after each timed block
* reported: mean and SD over the 5 repetitions (ms / image)
* also reported: batch-32 throughput (images / s) and Grad-CAM++ overhead for
  CovNet22 (one forward + one backward pass, batch size 1)

Weights are irrelevant for timing, so torchvision models are instantiated with
random weights and a 4-class head.

Hardware: NVIDIA GeForce RTX 3060 (8 GB)
Usage   : python 08_latency_benchmark.py
=============================================================================
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("defev", HERE / "07_definitive_covnet22_eval.py")
defev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(defev)

DEVICE = torch.device("cuda")
OUT = Path(os.environ.get("MRI_OUT_ROOT", r"C:\MRIData\results_v2")) / "latency_benchmark.csv"
NC = 4


def build(name):
    if name == "CovNet22":
        return defev.CovNet22(NC)
    if name == "ResNet-50":
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, NC); return m
    if name == "EfficientNet-B0":
        m = tvm.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, NC); return m
    if name == "MobileNetV2":
        m = tvm.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, NC); return m
    if name == "DenseNet-121":
        m = tvm.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, NC); return m
    if name == "VGG-16":
        m = tvm.vgg16(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, NC); return m
    if name == "Swin-T":
        m = tvm.swin_t(weights=None); m.head = nn.Linear(m.head.in_features, NC); return m
    if name == "ViT-B/16":
        m = tvm.vit_b_16(weights=None); m.heads.head = nn.Linear(m.heads.head.in_features, NC); return m
    raise ValueError(name)


@torch.no_grad()
def time_forward(model, bs, warm=50, n=200, reps=5):
    x = torch.randn(bs, 3, 224, 224, device=DEVICE)
    for _ in range(warm):
        model(x)
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / n * 1000)
    return np.array(out)


def time_gradcampp(model, warm=20, n=100, reps=5):
    """Forward + backward to the last conv block (cost of one Grad-CAM++ map)."""
    acts = {}
    layer = model.block4[0]
    layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    x = torch.randn(1, 3, 224, 224, device=DEVICE)

    def one():
        logits = model(x)
        a = acts["a"]
        g = torch.autograd.grad(logits[0, logits[0].argmax()], a)[0]
        g2, g3 = g ** 2, g ** 3
        alpha = g2 / (2 * g2 + (a * g3).sum((2, 3), keepdim=True) + 1e-8)
        w = (alpha * torch.relu(g)).sum((2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(1))
        return cam

    for _ in range(warm):
        one()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            one()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / n * 1000)
    return np.array(out)


def main():
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    rows = []
    for name in ["CovNet22", "ResNet-50", "EfficientNet-B0", "MobileNetV2",
                 "DenseNet-121", "VGG-16", "Swin-T", "ViT-B/16"]:
        torch.manual_seed(0)
        m = build(name).to(DEVICE).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        l1 = time_forward(m, 1)
        l32 = time_forward(m, 32, warm=10, n=30, reps=3)
        thr = 32 / (l32.mean() / 1000)
        row = dict(model=name, params_M=round(params, 2),
                   latency_bs1_ms_mean=round(l1.mean(), 3), latency_bs1_ms_sd=round(l1.std(ddof=1), 3),
                   throughput_bs32_img_s=round(thr, 1))
        if name == "CovNet22":
            gc = time_gradcampp(m)
            row["gradcampp_ms_mean"] = round(gc.mean(), 3)
            row["gradcampp_ms_sd"] = round(gc.std(ddof=1), 3)
        print(row)
        rows.append(row)
        del m
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
