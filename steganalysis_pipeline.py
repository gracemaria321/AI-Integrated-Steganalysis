#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steganalysis Pipeline implementing:
- Data Collection Layer
- Preprocessing Layer
- Testing & Processing Layer (Traditional + AI-enhanced)
- Postprocessing Layer (fusion + calibration)
- Evaluation Layer (Accuracy, Precision, Recall, F1)

Author: Your Name (Grace Maria et al.)
"""

import os
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image


# ---------------------------
# Utilities & Shared Types
# ---------------------------

SUPPORTED_EXT = (".jpg", ".jpeg", ".png")

@dataclass
class ImageRecord:
    path: str
    label: Optional[int] = None  # 0=cover, 1=stego (if known)
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class PreprocessedSample:
    record: ImageRecord
    img_gray: np.ndarray  # float32, normalized [0,1], shape (H, W)
    quality: Dict[str, float]   # e.g., laplacian_var, noise_energy
    meta: Dict[str, str]        # e.g., format, size_kb


# ============================================================
# 1) Data Collection Layer
# ============================================================

class DataCollectionLayer:
    """
    Collects JPEG and PNG images from directory structure.
    Optional labeling by folder name: {root}/cover/*.png, {root}/stego/*.jpg
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def collect(self) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        for ext in SUPPORTED_EXT:
            for path in glob.glob(os.path.join(self.root_dir, "**", f"*{ext}"), recursive=True):
                label = None
                # simple convention-based labeling
                lower = path.lower()
                if "cover" in lower:
                    label = 0
                elif "stego" in lower:
                    label = 1
                stat = os.stat(path)
                try:
                    with Image.open(path) as im:
                        fmt = im.format
                        w, h = im.size
                except Exception:
                    fmt, w, h = None, None, None

                records.append(ImageRecord(
                    path=path,
                    label=label,
                    format=fmt,
                    size_bytes=stat.st_size,
                    width=w,
                    height=h
                ))
        return records


# ============================================================
# 2) Preprocessing Layer
# ============================================================

class PreprocessingLayer:
    """
    - Verifies image type (JPEG/PNG)
    - Checks image size (min/maximum dims)
    - Computes simple quality metrics:
        * Laplacian variance (blur estimate)
        * High-pass noise energy
    - Converts to grayscale float in [0,1] and resizes to target_size
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 min_dim: int = 64, max_dim: int = 4096):
        self.target_size = target_size
        self.min_dim = min_dim
        self.max_dim = max_dim

    @staticmethod
    def to_grayscale(im: Image.Image) -> np.ndarray:
        arr = np.array(im.convert("L"), dtype=np.float32) / 255.0
        return arr

    @staticmethod
    def laplacian_variance(img: np.ndarray) -> float:
        # simple 3x3 Laplacian kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
        # pad and convolve
        pad = np.pad(img, 1, mode='reflect')
        H, W = img.shape
        out = np.zeros_like(img)
        for i in range(H):
            for j in range(W):
                region = pad[i:i+3, j:j+3]
                out[i, j] = np.sum(region * kernel)
        return float(np.var(out))

    @staticmethod
    def highpass_energy(img: np.ndarray) -> float:
        # use a simple high-pass filter (Sobel-like)
        hx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        hy = hx.T
        pad = np.pad(img, 1, mode='reflect')
        H, W = img.shape
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        for i in range(H):
            for j in range(W):
                region = pad[i:i+3, j:j+3]
                gx[i, j] = np.sum(region * hx)
                gy[i, j] = np.sum(region * hy)
        mag = np.sqrt(gx**2 + gy**2)
        return float(np.mean(mag))

    def preprocess(self, records: List[ImageRecord]) -> List[PreprocessedSample]:
        samples: List[PreprocessedSample] = []
        for r in records:
            if r.format not in ("JPEG", "PNG"):
                # skip unsupported or unreadable
                continue
            if r.width is None or r.height is None:
                continue
            if min(r.width, r.height) < self.min_dim or max(r.width, r.height) > self.max_dim:
                continue

            try:
                with Image.open(r.path) as im:
                    im = im.convert("RGB")
                    im = im.resize(self.target_size, resample=Image.Resampling.LANCZOS)
                    gray = self.to_grayscale(im)

                    q_lap = self.laplacian_variance(gray)
                    q_hp = self.highpass_energy(gray)
                    meta = {
                        "format": r.format,
                        "size_kb": f"{(r.size_bytes or 0) / 1024:.1f}",
                        "shape": f"{gray.shape[0]}x{gray.shape[1]}"
                    }
                    samples.append(PreprocessedSample(
                        record=r,
                        img_gray=gray,
                        quality={"laplacian_var": q_lap, "noise_energy": q_hp},
                        meta=meta
                    ))
            except Exception:
                # skip corrupted files
                continue
        return samples


# ============================================================
# 3) Testing & Processing Layer
#    (Traditional & AI-enhanced)
# ============================================================

class TraditionalStegAnalyzer:
    """
    Implements simple traditional heuristics:
    - LSB Chi-square heuristic on grayscale values
    - SPAM-like co-occurrence features of residuals (H and V)
    Returns feature vector and heuristic scores.
    """

    @staticmethod
    def chi_square_lsb(img: np.ndarray, bins: int = 128) -> float:
        """
        Chi-square test on the distribution of even/odd pixel values.
        High values may indicate LSB embedding.
        """
        # scale to 0..255 integers
        vals = (img * 255.0).astype(np.uint8).flatten()
        even = (vals % 2 == 0).astype(np.int32)
        odd = 1 - even

        # histogram by intensity bucket
        hist_even = np.zeros(bins, dtype=np.float64)
        hist_odd = np.zeros(bins, dtype=np.float64)
        bucket = (vals * bins / 256).astype(np.int32)
        for b, e, o in zip(bucket, even, odd):
            hist_even[b] += e
            hist_odd[b] += o

        expected = (hist_even + hist_odd) / 2.0
        # chi-square sum over bins where expected>0
        valid = expected > 0
        chi = np.sum(((hist_even[valid] - expected[valid]) ** 2) / expected[valid]) + \
              np.sum(((hist_odd[valid] - expected[valid]) ** 2) / expected[valid])
        return float(chi)

    @staticmethod
    def spam_like_features(img: np.ndarray, T: int = 2) -> np.ndarray:
        """
        SPAM-like residual co-occurrence: differences between neighbors
        clipped to [-T, T], then co-occurrence counts of (d_i, d_{i+1}).
        We compute horizontal and vertical co-occurrences and aggregate.
        """
        H, W = img.shape
        # integer residuals on scaled 0..255
        z = (img * 255.0).astype(np.int16)
        dx = z[:, 1:] - z[:, :-1]
        dy = z[1:, :] - z[:-1, :]
        dx = np.clip(dx, -T, T)
        dy = np.clip(dy, -T, T)

        def cooc(d):
            # pairs along axis
            pairs = np.stack([d[:, :-1], d[:, 1:]], axis=-1)  # shape (H, W-2, 2)
            offset = T
            size = 2 * T + 1
            M = np.zeros((size, size), dtype=np.float64)
            for i in range(pairs.shape[0]):
                for j in range(pairs.shape[1]):
                    a = pairs[i, j, 0] + offset
                    b = pairs[i, j, 1] + offset
                    M[a, b] += 1
            M /= np.sum(M) + 1e-9
            return M.flatten()

        f_h = cooc(dx)
        f_v = cooc(dy)
        features = np.concatenate([f_h, f_v], axis=0)
        return features.astype(np.float32)

    def extract_features(self, sample: PreprocessedSample) -> Dict[str, np.ndarray]:
        chi = self.chi_square_lsb(sample.img_gray)
        spam = self.spam_like_features(sample.img_gray, T=2)
        q = np.array([sample.quality["laplacian_var"],
                      sample.quality["noise_energy"]], dtype=np.float32)
        return {
            "chi_square": np.array([chi], dtype=np.float32),
            "spam": spam,
            "quality": q
        }


class AIEnhancedStegModel:
    """
    Lightweight logistic regression (numpy) over engineered features.
    - fit(X, y): trains weights via gradient descent
    - predict_proba(X): sigmoid(W·X + b)
    """

    def __init__(self, lr: float = 0.05, epochs: int = 200, l2: float = 1e-4):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.W: Optional[np.ndarray] = None
        self.b: float = 0.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        n, d = X.shape
        self.W = np.zeros(d, dtype=np.float32)
        self.b = 0.0
        for ep in range(self.epochs):
            z = X @ self.W + self.b
            p = self.sigmoid(z)
            # gradient
            grad_W = (X.T @ (p - y)) / n + self.l2 * self.W
            grad_b = float(np.mean(p - y))
            # update
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if verbose and (ep % max(1, self.epochs // 10) == 0 or ep == self.epochs - 1):
                loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
                print(f"[AIEnhancedStegModel] epoch={ep+1}/{self.epochs} loss={loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.W + self.b
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)


class TestingProcessingLayer:
    """
    Orchestrates feature extraction (Traditional) and AI model training/testing.
    - build_feature_matrix(samples): returns X, y (if labels exist)
    - train_ai_model(X_train, y_train): fits logistic regression
    - test(samples): returns per-sample dict of scores
    """

    def __init__(self):
        self.trad = TraditionalStegAnalyzer()
        self.ai = AIEnhancedStegModel()
        self.feature_names: List[str] = []

    def build_feature_matrix(self, samples: List[PreprocessedSample]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        rows = []
        labels = []
        for s in samples:
            feats = self.trad.extract_features(s)
            # concatenate: chi_square(1) + spam( (2T+1)^2*2 ) + quality(2)
            row = np.concatenate([feats["chi_square"], feats["spam"], feats["quality"]], axis=0)
            rows.append(row)
            if s.record.label is not None:
                labels.append(s.record.label)
        X = np.vstack(rows).astype(np.float32)
        y = np.array(labels, dtype=np.float32) if len(labels) == len(rows) else None

        # record feature names (sizes for documentation)
        self.feature_names = [
            "chi_square",
            f"spam_len={len(feats['spam'])}",
            "quality_laplacian_var",
            "quality_noise_energy"
        ]
        return X, y

    def train_ai_model(self, X_train: np.ndarray, y_train: np.ndarray):
        self.ai.fit(X_train, y_train, verbose=True)

    def test(self, samples: List[PreprocessedSample], threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
        results = {}
        # build features (no labels required)
        X, _ = self.build_feature_matrix(samples)
        ai_probs = self.ai.predict_proba(X)
        preds = (ai_probs >= threshold).astype(np.int32)
        for i, s in enumerate(samples):
            feats = self.trad.extract_features(s)
            results[s.record.path] = {
                "ai_prob_stego": float(ai_probs[i]),
                "ai_pred": int(preds[i]),
                "chi_square": float(feats["chi_square"][0]),
                "noise_energy": float(feats["quality"][1]),
                "laplacian_var": float(feats["quality"][0]),
                "label": int(s.record.label) if s.record.label is not None else -1
            }
        return results


# ============================================================
# 4) Postprocessing Layer (Fusion & Calibration)
# ============================================================

class PostprocessingLayer:
    """
    Combines traditional heuristic (chi-square) with AI probability via weighted fusion.
    Provides calibrated decision threshold using a small validation split.
    """

    def __init__(self, w_ai: float = 0.7, w_chi: float = 0.3):
        self.w_ai = w_ai
        self.w_chi = w_chi
        self.threshold = 0.5

    @staticmethod
    def _normalize(value: float, eps: float = 1e-9) -> float:
        # log normalization for chi-square magnitude
        return float(1.0 - math.exp(-value / 10.0 + eps))

    def calibrate(self, probs: np.ndarray, labels: np.ndarray):
        """
        Simple calibration: choose threshold maximizing F1 on validation set.
        """
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            preds = (probs >= t).astype(np.int32)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t

    def fuse(self, ai_prob: float, chi_value: float) -> float:
        chi_norm = self._normalize(chi_value)
        return float(self.w_ai * ai_prob + self.w_chi * chi_norm)


# ============================================================
# 5) Evaluation Layer
# ============================================================

class EvaluationLayer:
    """
    Computes Accuracy, Precision, Recall, F1-score and Confusion Matrix.
    """

    @staticmethod
    def metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        tp = float(np.sum((preds == 1) & (labels == 1)))
        tn = float(np.sum((preds == 0) & (labels == 0)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    @staticmethod
    def print_report(res: Dict[str, float]):
        print("\n=== Evaluation Report ===")
        print(f"Accuracy : {res['accuracy']:.4f}")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Recall   : {res['recall']:.4f}")
        print(f"F1-score : {res['f1']:.4f}")
        print(f"Confusion: TP={int(res['tp'])} TN={int(res['tn'])} FP={int(res['fp'])} FN={int(res['fn'])}")


# ============================================================
# Example Orchestration
# ============================================================

def train_test_pipeline(root_dir: str,
                        val_fraction: float = 0.2,
                        seed: int = 42):
    """
    Orchestrates the entire framework using files under root_dir.
    Expects labeling via folder names (contains 'cover' or 'stego').
    """

    random.seed(seed); np.random.seed(seed)

    # 1) Data Collection
    collector = DataCollectionLayer(root_dir)
    records = collector.collect()
    print(f"[DataCollection] Found {len(records)} images.")

    # Keep only labeled samples for training/evaluation
    labeled = [r for r in records if r.label is not None]
    if len(labeled) < 10:
        print("[Warning] Not enough labeled samples. "
              "Add images under .../cover/ and .../stego/ to enable training.")
    # 2) Preprocessing
    prep = PreprocessingLayer(target_size=(256, 256))
    samples = prep.preprocess(labeled)
    print(f"[Preprocessing] Kept {len(samples)} samples after quality/type/size checks.")

    # 3) Testing & Processing: build features
    tp_layer = TestingProcessingLayer()
    X, y = tp_layer.build_feature_matrix(samples)
    print(f"[Features] X shape={X.shape}, y len={len(y) if y is not None else 0}")

    # train/val split
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # train AI-enhanced logistic model
    tp_layer.train_ai_model(X_train, y_train)
    ai_val_probs = tp_layer.ai.predict_proba(X_val)

    # 4) Postprocessing: calibrate threshold & fuse
    post = PostprocessingLayer(w_ai=0.7, w_chi=0.3)
    post.calibrate(ai_val_probs, y_val)
    print(f"[Postprocessing] Calibrated threshold={post.threshold:.3f}")

    # Apply to the whole set
    results = tp_layer.test(samples, threshold=post.threshold)

    # Build fused predictions and evaluate
    fused_probs = []
    labels = []
    for path, r in results.items():
        fused = post.fuse(r["ai_prob_stego"], r["chi_square"])
        fused_probs.append(fused)
        labels.append(r["label"])
    fused_probs = np.array(fused_probs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    preds = (fused_probs >= post.threshold).astype(np.int32)
    eval_layer = EvaluationLayer()
    report = eval_layer.metrics(preds, labels)
    eval_layer.print_report(report)

    # Optional: return artifacts for further analysis
    return {
        "records": records,
        "samples": samples,
        "features": (X, y),
        "results": results,
        "fused_probs": fused_probs,
        "labels": labels,
        "report": report
    }


if __name__ == "__main__":
    # Example usage:
    # Arrange your data as:
    # dataset_root/
    #   cover/  ... (JPEG/PNG)
    #   stego/  ... (JPEG/PNG)
    #
    # Then run:
    # python steganalysis_pipeline.py
    #
    root = "./dataset_root"
    artifacts = train_test_pipeline(root)
