
# -*- coding: utf-8 -*-
"""
Updated ROC/PR plotting for steganalysis_pipeline artifacts.

- Fixes AUC deprecation (uses np.trapezoid)
- Adds percentile-bounded threshold sweep to avoid degenerate curves
- Plots ROC (AI-only vs Fusion) and PR curves
- Generates saliency-style residual & LSB visualizations

Author: Grace Maria
"""

import numpy as np
import matplotlib.pyplot as plt

# 1) Import pipeline and run (or reuse existing artifacts if already computed)
from steganalysis_pipeline import train_test_pipeline

ROOT_DIR = "./dataset_root"   # <-- set your dataset path
artifacts = train_test_pipeline(ROOT_DIR)

# 2) Gather labels and probabilities (AI-only & Fusion) aligned with samples
labels = artifacts["labels"].astype(np.int32)
samples = artifacts["samples"]
results = artifacts["results"]

ai_probs   = np.array([results[s.record.path]["ai_prob_stego"] for s in samples], dtype=np.float32)
fused_probs = artifacts["fused_probs"].astype(np.float32)

# ---------------------------------------------------------------------
# Utility: percentile-bounded thresholds to avoid extremes
# ---------------------------------------------------------------------
def bounded_thresholds(scores: np.ndarray, n_points: int = 200, low_pct: float = 1.0, high_pct: float = 99.0):
    lo, hi = np.percentile(scores, [low_pct, high_pct])
    if lo == hi:   # degenerate case, fallback to [0,1]
        return np.linspace(0.0, 1.0, n_points)
    return np.linspace(lo, hi, n_points)

# ---------------------------------------------------------------------
# ROC computation (no sklearn)
# ---------------------------------------------------------------------
def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = bounded_thresholds(y_score, n_points=300, low_pct=1.0, high_pct=99.0)
    tpr, fpr = [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int32)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        tpr.append(TP / (TP + FN + 1e-9))
        fpr.append(FP / (FP + TN + 1e-9))
    return np.array(fpr), np.array(tpr)

def auc(x: np.ndarray, y: np.ndarray):
    # Sort by FPR to integrate correctly
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    return np.trapezoid(y_s, x_s)

# ---------------------------------------------------------------------
# PR computation (no sklearn)
# ---------------------------------------------------------------------
def pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = bounded_thresholds(y_score, n_points=300, low_pct=1.0, high_pct=99.0)
    precision, recall = [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int32)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        prec = TP / (TP + FP + 1e-9)
        rec  = TP / (TP + FN + 1e-9)
        precision.append(prec)
        recall.append(rec)
    return np.array(recall), np.array(precision)

# ---------------------------------------------------------------------
# 3) Compute curves: AI-only vs Fusion
# ---------------------------------------------------------------------
fpr_ai,  tpr_ai  = roc_curve(labels, ai_probs)
fpr_fus, tpr_fus = roc_curve(labels, fused_probs)
auc_ai   = auc(fpr_ai, tpr_ai)
auc_fus  = auc(fpr_fus, tpr_fus)

# ROC plot
# plt.figure(figsize=(7.0, 5.0), dpi=120)
# plt.plot(fpr_ai,  tpr_ai,  label=f"AI-only (AUC={auc_ai:.2f})", lw=2, color="#1f77b4")
# plt.plot(fpr_fus, tpr_fus, label=f"Fusion (AI + Chi-square) (AUC={auc_fus:.2f})", lw=2, color="#d62728")
# plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

# # plt.title("Figure 2. ROC Curves for AI vs Traditional Heuristic", fontsize=12)
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.grid(alpha=0.25)
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("fig2_roc_ai_vs_fusion.png")
# plt.show()

# ---------------------------------------------------------------------
# 4) Precision–Recall curves
# ---------------------------------------------------------------------
rec_ai,  prec_ai  = pr_curve(labels, ai_probs)
rec_fus, prec_fus = pr_curve(labels, fused_probs)

plt.figure(figsize=(7.0, 5.0), dpi=120)
plt.plot(rec_ai,  prec_ai,  label="AI-only", lw=2, color="#1f77b4")
plt.plot(rec_fus, prec_fus, label="Fusion (AI + Chi-square)", lw=2, color="#d62728")

# plt.title("Figure 3. Precision–Recall Curves at Low Payload", fontsize=12)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.grid(alpha=0.25)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("fig3_pr_ai_vs_fusion.png")
plt.show()

# ---------------------------------------------------------------------
# 5) Saliency-style visualization (Residual magnitude + LSB parity)
# ---------------------------------------------------------------------
def highpass_magnitude(img_gray: np.ndarray):
    hx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    hy = hx.T
    pad = np.pad(img_gray, 1, mode='reflect')
    H, W = img_gray.shape
    gx = np.zeros_like(img_gray, dtype=np.float32)
    gy = np.zeros_like(img_gray, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            region = pad[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * hx)
            gy[i, j] = np.sum(region * hy)
    return np.sqrt(gx**2 + gy**2)

def lsb_parity(img_gray: np.ndarray):
    z = (img_gray * 255.0).astype(np.uint8)
    return (z % 2).astype(np.uint8)

# Select first 3 samples for display
k = min(3, len(samples))
selected = samples[:k]

fig, axes = plt.subplots(k, 3, figsize=(9, 3*k), dpi=120)
if k == 1:
    axes = np.array([axes])  # keep 2D indexing

for idx, s in enumerate(selected):
    img = s.img_gray
    hp  = highpass_magnitude(img)
    lsb = lsb_parity(img)

    hp_vis = (hp - hp.min()) / (hp.max() - hp.min() + 1e-9)

    axes[idx, 0].imshow(img, cmap="gray")
    axes[idx, 0].set_title(f"Original (label={s.record.label})")
    axes[idx, 0].axis("off")

    im2 = axes[idx, 1].imshow(hp_vis, cmap="inferno")
    axes[idx, 1].set_title("Residual Magnitude (High-pass)")
    axes[idx, 1].axis("off")
    fig.colorbar(im2, ax=axes[idx, 1], fraction=0.046, pad=0.04)

    axes[idx, 2].imshow(lsb, cmap="gray")
    axes[idx, 2].set_title("LSB Parity Map (Even/Odd)")
    axes[idx, 2].axis("off")

# plt.suptitle("Figure 4. Saliency Visualization of Residual and LSB Indicators", fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig4_saliency_residual_lsb.png")
