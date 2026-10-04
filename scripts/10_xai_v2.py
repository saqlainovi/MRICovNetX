"""
=============================================================================
MRICovNetX - Experiment 10: Qualitative XAI with the definitive CovNet22 models
=============================================================================
Grad-CAM, multi-layer CAM with brain masking, Grad-CAM++ with deep-core
weighting / hemisphere gating / component filtering (Dataset-3), and LIME
superpixel explanations (Dataset-4), all computed on the seed-42 checkpoints
written by 07_definitive_covnet22_eval.py.

    python 10_xai_v2.py --out <overleaf>/figs/v2
=============================================================================
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import importlib
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
E = importlib.import_module("07_definitive_covnet22_eval")
DEV = E.DEVICE


def load_model(ds, n_cls):
    cfg = E.DATASETS[ds]
    m = E.CovNet22(n_cls, cfg["p_conv"], cfg["p_fc"]).to(DEV)
    m.load_state_dict(torch.load(E.OUT_ROOT / ds.replace("-", "") / "full_seed42" / "covnet22.pt", map_location=DEV))
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    return m.eval()


def cams(model, x_uint8, cls):
    """Return dict layer -> (gradcam, gradcam++) at native resolution."""
    layers = {"conv3": model.block2[3], "conv4": model.block3[1], "conv5": model.block4[1]}
    acts, grads, hs = {}, {}, []
    for k, mod in layers.items():
        def fwd(_m, _i, o, k=k):
            acts[k] = o
            o.register_hook(lambda g, k=k: grads.__setitem__(k, g))
        hs.append(mod.register_forward_hook(fwd))
    x = E.to_input(x_uint8.unsqueeze(0).to(DEV))
    out = model(x)
    model.zero_grad()
    out[0, cls].backward()
    for h in hs:
        h.remove()
    res = {}
    for k in layers:
        A, G = acts[k][0].detach(), grads[k][0].detach()
        gc = F.relu((G.mean((1, 2))[:, None, None] * A).sum(0))
        g2, g3 = G ** 2, G ** 3
        alpha = g2 / (2 * g2 + A.sum((1, 2), keepdim=True) * g3 + 1e-7)
        w = (alpha * F.relu(G)).sum((1, 2))
        gpp = F.relu((w[:, None, None] * A).sum(0))
        res[k] = (gc.cpu().numpy(), gpp.cpu().numpy())
    return res, torch.softmax(out, 1)[0].detach().cpu().numpy()


def up(c, size=224):
    c = cv2.resize(c.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    return (c - c.min()) / (c.max() - c.min() + 1e-8)


def brain_mask(gray):
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    ff = m.copy(); h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = m | cv2.bitwise_not(ff)
    d = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return (m > 0).astype(np.float32), d / (d.max() + 1e-8)


def basic(res):
    return up(res["conv5"][0])


def multilayer(res, mask, dist):
    c = 0.2 * up(res["conv3"][0]) + 0.3 * up(res["conv4"][0]) + 0.5 * up(res["conv5"][0])
    c = c * mask * dist ** 0.3
    return (c - c.min()) / (c.max() - c.min() + 1e-8)


def gradcampp_focused(res, mask, dist):
    c = 0.2 * up(res["conv3"][1]) + 0.3 * up(res["conv4"][1]) + 0.5 * up(res["conv5"][1])
    c = c * mask * dist ** 0.7                       # deep-core emphasis, rim suppression
    ys, xs = np.nonzero(mask)
    cx = int(xs.mean()) if len(xs) else 112          # hemisphere gating
    el, er = c[:, :cx].sum(), c[:, cx:].sum()
    if max(el, er) / (min(el, er) + 1e-8) > 1.5:
        if el < er:
            c[:, :cx] *= 0.4
        else:
            c[:, cx:] *= 0.4
    c = (c - c.min()) / (c.max() - c.min() + 1e-8)
    b = (c > 0.5).astype(np.uint8)                   # component filtering
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    if n > 1:
        keep = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.float32)
        keep = cv2.GaussianBlur(cv2.dilate(keep, np.ones((9, 9), np.uint8)), (21, 21), 0)
        c = c * (0.25 + 0.75 * keep / (keep.max() + 1e-8))
    return (c - c.min()) / (c.max() - c.min() + 1e-8)


def overlay(img_rgb, cam):
    col = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (0.55 * img_rgb + 0.45 * col).astype(np.uint8)


def pick(probs, y, labels, per_class, min_conf=0.95):
    pred = probs.argmax(1)
    out = []
    for c in range(len(labels)):
        idx = np.where((y == c) & (pred == c) & (probs[:, c] >= min_conf))[0]
        sel = idx[np.linspace(0, len(idx) - 1, per_class).astype(int)]
        out.extend((int(i), c) for i in sel)
    return out


def run_d3(out):
    ds = "Dataset-3"; labels = E.DATASETS[ds]["labels"]
    data = E.load_dataset(ds)
    model = load_model(ds, len(labels))
    probs, _ = E.predict(model, data["xte"])
    y = data["yte"]

    def explain(i, c):
        x = data["xte"][i]
        img = x.permute(1, 2, 0).numpy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask, dist = brain_mask(gray)
        res, p = cams(model, x, c)
        return img, res, mask, dist, p

    # Fig A: basic Grad-CAM, one sample per class
    sel = pick(probs, y, labels, 1)
    fig, ax = plt.subplots(len(sel), 3, figsize=(7.5, 2.5 * len(sel)))
    for r, (i, c) in enumerate(sel):
        img, res, mask, dist, p = explain(i, c)
        cam = basic(res)
        for a, im, t in zip(ax[r], [img, cam, overlay(img, cam)], ["Original", "Grad-CAM heatmap", "Overlay"]):
            a.imshow(im, cmap="jet" if im is cam else None); a.axis("off")
            a.set_title(f"{t}" + (f"\n{labels[c]} ({p[c]*100:.1f}%)" if t == "Original" else ""), fontsize=8)
    fig.savefig(out / "xai_basic_gradcam_v2.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Fig B/C/D: one glioma sample through the three techniques
    i, c = pick(probs, y, labels, 3)[1]
    img, res, mask, dist, p = explain(i, c)
    maps = {"Grad-CAM": basic(res), "Multi-layer CAM + brain mask": multilayer(res, mask, dist),
            "Grad-CAM++ (deep-core, gated)": gradcampp_focused(res, mask, dist)}
    for fname, key in [("xai_multilayer_v2.png", "Multi-layer CAM + brain mask"),
                       ("xai_gradcampp_v2.png", "Grad-CAM++ (deep-core, gated)")]:
        fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.6))
        for a, im, t in zip(ax, [img, maps[key], overlay(img, maps[key])], ["Original", key, "Overlay"]):
            a.imshow(im, cmap="jet" if im is maps[key] else None); a.axis("off"); a.set_title(t, fontsize=8)
        fig.savefig(out / fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for a, (t, m) in zip(ax, maps.items()):
        a.imshow(overlay(img, m)); a.axis("off"); a.set_title(t, fontsize=8)
    fig.savefig(out / "xai_comparison_v2.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Fig E: 12 tumour cases (4 per tumour class), Grad-CAM++ overlays
    sel = [s for s in pick(probs, y, labels, 4) if labels[s[1]] != "No tumor"]
    fig, ax = plt.subplots(3, 4, figsize=(9, 7))
    for a, (i, c) in zip(ax.ravel(), sel):
        img, res, mask, dist, p = explain(i, c)
        a.imshow(overlay(img, gradcampp_focused(res, mask, dist))); a.axis("off")
        a.set_title(f"{labels[c]} ({p[c]*100:.1f}%)", fontsize=8)
    fig.savefig(out / "xai_multisample_v2.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("D3 XAI done")


def superpixels(img, k=60, seed=0):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    feat = np.stack([xx / w * 2.0, yy / h * 2.0, gray / 255.0], -1).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.setRNGSeed(seed)
    _, lab, _ = cv2.kmeans(feat, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.reshape(h, w)
    seg = np.zeros_like(lab); nid = 0
    for v in np.unique(lab):                       # split disconnected clusters
        n, cc = cv2.connectedComponents((lab == v).astype(np.uint8))
        for j in range(1, n):
            seg[cc == j] = nid; nid += 1
    return seg, nid


@torch.no_grad()
def lime_explain(model, x_uint8, cls, n_samples=1000, seed=0):
    img = x_uint8.permute(1, 2, 0).numpy()
    seg, n = superpixels(img)
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, size=(n_samples, n)); Z[0] = 1
    blur = cv2.GaussianBlur(img, (0, 0), 8)
    preds = []
    for b in range(0, n_samples, 64):
        batch = []
        for z in Z[b:b + 64]:
            keep = z[seg].astype(bool)[:, :, None]
            batch.append(np.where(keep, img, blur))
        xb = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(DEV)
        preds.append(torch.softmax(model(E.to_input(xb)), 1)[:, cls].cpu().numpy())
    yv = np.concatenate(preds)
    dist = 1 - (Z @ Z[0]) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(Z[0]) + 1e-8)
    wts = np.exp(-(dist ** 2) / 0.25 ** 2)
    Xd = np.hstack([Z, np.ones((n_samples, 1))])
    W = np.diag(wts)
    coef = np.linalg.solve(Xd.T @ W @ Xd + 1.0 * np.eye(n + 1), Xd.T @ W @ yv)[:n]
    return img, seg, coef


def run_d4(out):
    ds = "Dataset-4"; labels = E.DATASETS[ds]["labels"]
    data = E.load_dataset(ds)
    model = load_model(ds, len(labels))
    probs, _ = E.predict(model, data["xte"])
    sel = pick(probs, data["yte"], labels, 1)
    fig, ax = plt.subplots(len(sel), 3, figsize=(7.5, 2.6 * len(sel)))
    for r, (i, c) in enumerate(sel):
        img, seg, coef = lime_explain(model, data["xte"][i], c)
        top = np.argsort(coef)[::-1][:5]
        m = np.isin(seg, top)
        hl = img.copy(); hl[~m] = (hl[~m] * 0.35).astype(np.uint8)
        edges = cv2.Canny(m.astype(np.uint8) * 255, 50, 150) > 0
        hl[edges] = [255, 255, 0]
        wmap = coef[seg]; v = np.abs(wmap).max() + 1e-8
        ax[r, 0].imshow(img); ax[r, 0].set_title(f"Original\n{labels[c]} ({probs[i, c]*100:.1f}%)", fontsize=8)
        ax[r, 1].imshow(hl); ax[r, 1].set_title("Top-5 supporting superpixels", fontsize=8)
        ax[r, 2].imshow(wmap, cmap="RdBu_r", vmin=-v, vmax=v); ax[r, 2].set_title("LIME weights (red = supports)", fontsize=8)
        for a in ax[r]:
            a.axis("off")
    fig.savefig(out / "lime_d4_v2.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("D4 LIME done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    o = Path(ap.parse_args().out); o.mkdir(parents=True, exist_ok=True)
    run_d3(o); run_d4(o)
