"""
DAC-RViT core library.

Shared components used by both the training pipeline (dac_rvit_train.py)
and the inference package (inference/predict.py, inference/evaluate_test.py):

  * dataset discovery and block-wise train/val/test splitting
  * image/numeric preprocessing
  * Hybrid ResNet-18 + Vision Transformer classifier (Phase 1)
  * per-bin XGBoost regressors on (cos, sin) targets (Phase 2)
  * Confidence Estimation Pipeline (temperature scaling, meta-classifier, threshold)
  * circular angle utilities and evaluation metrics
  * a single `DACRViTPredictor` class that performs end-to-end inference

All angles are in degrees in the interval [-180, 180).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

try:
    import xgboost as xgb
except ImportError as _e:  # pragma: no cover
    xgb = None

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import joblib


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Positional numerical features (PNFs): local origin (LO) and tip (TD) image
# coordinates. The fiducial-sphere columns present in labels.csv are never read.
PNF_COLUMNS = ["X_Local_Origin", "Y_Local_Origin",
               "X_Tip_Displacement", "Y_Tip_Displacement"]
LABEL_COLUMN = "Roll_Angle"
FRAME_COLUMN = "frame"
TIME_COLUMN = "Time (ms)"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs and enable deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Circular angle utilities
# ----------------------------------------------------------------------------

def wrap_deg(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def circular_error_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Shortest-arc absolute angular difference, in [0, 180]."""
    d = np.deg2rad(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64))
    return np.abs(np.rad2deg(np.arctan2(np.sin(d), np.cos(d))))


def angle_to_cossin(angle_deg: np.ndarray) -> np.ndarray:
    """Map angle (deg) to unit-vector components [cos, sin]."""
    r = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    return np.stack([np.cos(r), np.sin(r)], axis=-1)


def cossin_to_angle(tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """Recover angle (deg, [-180,180)) from [cos, sin] via two-argument atan2."""
    return wrap_deg(np.rad2deg(np.arctan2(ty, tx)))


def circular_mean_deg(a: np.ndarray) -> float:
    r = np.deg2rad(np.asarray(a, dtype=np.float64))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


def circular_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Circular coefficient of determination:
        1 - sum(d(pred,true)^2) / sum(d(true, circ_mean(true))^2)
    where d is the shortest-arc distance. Invariant to the angular origin.
    """
    true = np.asarray(true, dtype=np.float64)
    res = circular_error_deg(pred, true) ** 2
    tot = circular_error_deg(np.full_like(true, circular_mean_deg(true)), true) ** 2
    return float(1.0 - res.sum() / max(tot.sum(), 1e-12))


def circular_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    """Jammalamadaka-Sarma circular correlation coefficient."""
    a = np.deg2rad(np.asarray(pred, dtype=np.float64))
    b = np.deg2rad(np.asarray(true, dtype=np.float64))
    am = np.deg2rad(circular_mean_deg(pred))
    bm = np.deg2rad(circular_mean_deg(true))
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    den = np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))
    return float(num / max(den, 1e-12))


# ----------------------------------------------------------------------------
# Angular bins
# ----------------------------------------------------------------------------

@dataclass
class BinScheme:
    """N equal-width angular bins covering [-180, 180)."""
    n_bins: int

    @property
    def width(self) -> float:
        return 360.0 / self.n_bins

    def index(self, angle_deg: np.ndarray) -> np.ndarray:
        a = wrap_deg(angle_deg)
        idx = np.floor((a + 180.0) / self.width).astype(int)
        return np.clip(idx, 0, self.n_bins - 1)

    def lower(self, i: int) -> float:
        return -180.0 + i * self.width

    def upper(self, i: int) -> float:
        return self.lower(i) + self.width

    def centre(self, i: int) -> float:
        return wrap_deg(self.lower(i) + self.width / 2.0)

    def contains(self, angle_deg: float, i: int) -> bool:
        """True if the shortest-arc distance to the bin centre is within half a width."""
        return circular_error_deg(np.array([angle_deg]), np.array([self.centre(i)]))[0] <= self.width / 2.0 + 1e-9

    def centres(self) -> np.ndarray:
        return np.array([self.centre(i) for i in range(self.n_bins)])


# ----------------------------------------------------------------------------
# Dataset discovery and splitting
# ----------------------------------------------------------------------------

def discover_frames(main_folder: str) -> pd.DataFrame:
    """
    Scan `main_folder` for recording subfolders. Each subfolder must contain a
    labels.csv and frames named frame_XXXX.png. Returns one row per frame with
    columns: recording, frame_index, path, roll, pnf_0..pnf_3, time_ms.
    Frames are ordered by frame_index within each recording.
    """
    rows = []
    for rec in sorted(os.listdir(main_folder)):
        folder = os.path.join(main_folder, rec)
        csv_path = os.path.join(folder, "labels.csv")
        if not os.path.isdir(folder) or not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        required = [FRAME_COLUMN, LABEL_COLUMN] + PNF_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        for _, r in df.iterrows():
            name = str(r[FRAME_COLUMN]).strip()
            path = os.path.join(folder, name)
            if not (name.startswith("frame_") and name.endswith(".png") and os.path.exists(path)):
                continue
            try:
                fidx = int(name[len("frame_"):-len(".png")])
            except ValueError:
                continue
            rows.append({
                "recording": rec,
                "frame_index": fidx,
                "path": path,
                "roll": float(r[LABEL_COLUMN]),
                "pnf_0": float(r[PNF_COLUMNS[0]]),
                "pnf_1": float(r[PNF_COLUMNS[1]]),
                "pnf_2": float(r[PNF_COLUMNS[2]]),
                "pnf_3": float(r[PNF_COLUMNS[3]]),
                "time_ms": float(r[TIME_COLUMN]) if TIME_COLUMN in df.columns else float("nan"),
            })
    if not rows:
        raise FileNotFoundError(f"No frames found under {main_folder}")
    out = pd.DataFrame(rows).sort_values(["recording", "frame_index"]).reset_index(drop=True)
    out["roll"] = wrap_deg(out["roll"].values)
    return out


def assign_blocks(df: pd.DataFrame, block_len: int) -> pd.DataFrame:
    """
    Assign each frame to a contiguous block of `block_len` consecutive frames
    within its recording. Block ids are globally unique strings
    '<recording>#<block_number>'.
    """
    df = df.copy()
    df["block_number"] = df.groupby("recording").cumcount() // block_len
    df["block_id"] = df["recording"] + "#" + df["block_number"].astype(str)
    return df


def subsample_blocks(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """
    Keep a random `fraction` of blocks from every recording (at least one block
    per recording). Used to run the full pipeline on a small subset of data.
    """
    if fraction >= 1.0:
        return df
    rng = np.random.RandomState(seed)
    keep = []
    for rec, g in df.groupby("recording"):
        blocks = g["block_id"].unique()
        n_keep = max(1, int(round(len(blocks) * fraction)))
        keep.extend(rng.choice(blocks, size=n_keep, replace=False).tolist())
    return df[df["block_id"].isin(keep)].reset_index(drop=True)


def block_split(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int) -> pd.DataFrame:
    """
    Split frames into train/val/test by whole blocks. Blocks are shuffled with
    `seed` and assigned greedily until the cumulative frame count reaches the
    requested fractions. Adds a 'split' column.
    """
    rng = np.random.RandomState(seed)
    blocks = np.array(df["block_id"].unique(), dtype=object)
    rng.shuffle(blocks)
    sizes = df.groupby("block_id").size()
    total = len(df)
    n_train_target = train_frac * total
    n_val_target = (train_frac + val_frac) * total
    split_of_block = {}
    cum = 0
    for b in blocks:
        if cum < n_train_target:
            split_of_block[b] = "train"
        elif cum < n_val_target:
            split_of_block[b] = "val"
        else:
            split_of_block[b] = "test"
        cum += sizes[b]
    df = df.copy()
    df["split"] = df["block_id"].map(split_of_block)
    return df


def build_manifest(main_folder: str, block_len: int, train_frac: float, val_frac: float,
                   seed: int, data_fraction: float = 1.0) -> pd.DataFrame:
    """Discover frames, assign blocks, optionally subsample, and split."""
    df = discover_frames(main_folder)
    df = assign_blocks(df, block_len)
    df = subsample_blocks(df, data_fraction, seed)
    df = block_split(df, train_frac, val_frac, seed)
    return df


def boundary_adjacency_stats(manifest: pd.DataFrame) -> Dict[str, float]:
    """
    Fraction of test frames that are immediately adjacent (one frame) to a
    training frame within the same recording. Reported for transparency about
    block-boundary proximity.
    """
    adjacent = 0
    n_test = 0
    for _, g in manifest.groupby("recording"):
        g = g.sort_values("frame_index")
        splits = g["split"].values
        for i, s in enumerate(splits):
            if s != "test":
                continue
            n_test += 1
            left = splits[i - 1] if i > 0 else None
            right = splits[i + 1] if i + 1 < len(splits) else None
            if left == "train" or right == "train":
                adjacent += 1
    return {"n_test": n_test,
            "n_test_adjacent_to_train": adjacent,
            "pct_test_adjacent_to_train": 100.0 * adjacent / max(n_test, 1)}


# ----------------------------------------------------------------------------
# Image and numeric preprocessing
# ----------------------------------------------------------------------------

@dataclass
class NumericNormalizer:
    """Standardises the 4 PNFs with statistics computed on the training split."""
    mean: List[float]
    std: List[float]

    @classmethod
    def fit(cls, pnf: np.ndarray) -> "NumericNormalizer":
        m = pnf.mean(axis=0)
        s = np.maximum(pnf.std(axis=0), 1e-6)
        return cls(mean=m.tolist(), std=s.tolist())

    def __call__(self, pnf: np.ndarray) -> np.ndarray:
        return (np.asarray(pnf, dtype=np.float32) - np.array(self.mean, dtype=np.float32)) / \
               np.array(self.std, dtype=np.float32)


def load_image_tensor(path: str, image_size: int) -> torch.Tensor:
    """
    Load a grayscale fluoroscopy crop, replicate to 3 channels, resize and
    return a uint8 tensor of shape (3, H, W). ImageNet normalisation is applied
    later, at batch time.
    """
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1)


def normalize_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,H,W) -> float normalised with ImageNet statistics."""
    x = x_uint8.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def photometric_augment(x_uint8: torch.Tensor, cfg: Dict) -> torch.Tensor:
    """
    Photometric-only augmentation applied to a uint8 batch. No geometric
    transform is used so that image content, PNFs and roll labels remain
    mutually consistent.
    """
    x = x_uint8.float() / 255.0
    b = x.shape[0]
    dev = x.device
    if cfg.get("brightness", 0) > 0:
        f = 1.0 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * cfg["brightness"]
        x = x * f
    if cfg.get("contrast", 0) > 0:
        f = 1.0 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * cfg["contrast"]
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * f + m
    if cfg.get("gaussian_noise_std", 0) > 0:
        x = x + torch.randn_like(x) * cfg["gaussian_noise_std"]
    if cfg.get("blur_prob", 0) > 0:
        k = int(cfg.get("blur_kernel", 3)) | 1
        sigma = float(cfg.get("blur_sigma", 0.8))
        ax = torch.arange(k, device=dev, dtype=torch.float32) - (k - 1) / 2
        g1 = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g1 = g1 / g1.sum()
        kernel = (g1[:, None] * g1[None, :]).expand(3, 1, k, k)
        blurred = F.conv2d(x, kernel, padding=k // 2, groups=3)
        mask = (torch.rand(b, 1, 1, 1, device=dev) < cfg["blur_prob"]).float()
        x = mask * blurred + (1 - mask) * x
    if cfg.get("erase_prob", 0) > 0:
        # Random erasing: blank one rectangle per image (label-safe, no geometry change)
        _, _, hgt, wid = x.shape
        for j in range(b):
            if torch.rand(1).item() < cfg["erase_prob"]:
                eh = int(hgt * (0.05 + 0.15 * torch.rand(1).item()))
                ew = int(wid * (0.05 + 0.15 * torch.rand(1).item()))
                y0 = int(torch.randint(0, max(hgt - eh, 1), (1,)).item())
                x0 = int(torch.randint(0, max(wid - ew, 1), (1,)).item())
                x[j, :, y0:y0 + eh, x0:x0 + ew] = x[j].mean()
    x = x.clamp(0, 1)
    return (x * 255.0).round().to(torch.uint8)


# ----------------------------------------------------------------------------
# Phase 1 model: Hybrid ResNet-18 + Vision Transformer
# ----------------------------------------------------------------------------

class TokenTransformer(nn.Module):
    """Linear token projection + learnable positional embedding + Transformer encoder."""

    def __init__(self, in_features: int, num_tokens: int, embed_dim: int,
                 num_heads: int, num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=ff_dim, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (B, C, H, W) -> tokens (B, H*W, C)
        b, c, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        x = self.proj(tokens) + self.pos[:, : h * w]
        x = self.encoder(x)
        return self.norm(x)


class HybridRViT(nn.Module):
    """
    Phase 1 classifier.

    input_mode:
      'both'  : image stream (ResNet-18 -> ViT -> GAP) + numeric stream (MLP)
      'image' : image stream only
      'pnf'   : numeric stream only (no image backbone)

    `embedding()` returns the feature vector used by the Phase 2 regressors:
    the image-stream embedding when available, otherwise the numeric embedding.
    """

    def __init__(self, num_classes: int, embed_dim: int = 300, num_heads: int = 10,
                 num_layers: int = 10, ff_dim: int = 1200, dropout: float = 0.3,
                 image_size: int = 224, input_mode: str = "both",
                 pretrained_backbone: bool = True):
        super().__init__()
        assert input_mode in ("both", "image", "pnf")
        self.input_mode = input_mode
        self.embed_dim = embed_dim
        self.use_image = input_mode in ("both", "image")
        self.use_pnf = input_mode in ("both", "pnf")

        if self.use_image:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            resnet = models.resnet18(weights=weights)
            # Keep layers up to the last convolutional stage (stride 32 -> 7x7 map for 224 input)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.fmap_hw = image_size // 32
            self.vit = TokenTransformer(resnet.fc.in_features, self.fmap_hw ** 2,
                                        embed_dim, num_heads, num_layers, ff_dim, dropout)
        if self.use_pnf:
            self.numeric_mlp = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, embed_dim))

        fused = embed_dim * (int(self.use_image) + int(self.use_pnf))
        self.head = nn.Linear(fused, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _streams(self, x: Optional[torch.Tensor], pnf: Optional[torch.Tensor]):
        feats = []
        img_emb = None
        if self.use_image:
            img_emb = self.vit(self.backbone(x)).mean(dim=1)
            feats.append(img_emb)
        if self.use_pnf:
            feats.append(self.numeric_mlp(pnf))
        return torch.cat(feats, dim=1), img_emb

    def forward(self, x: Optional[torch.Tensor], pnf: Optional[torch.Tensor]) -> torch.Tensor:
        fused, _ = self._streams(x, pnf)
        return self.head(fused)

    def embedding(self, x: Optional[torch.Tensor], pnf: Optional[torch.Tensor]) -> torch.Tensor:
        fused, img_emb = self._streams(x, pnf)
        return img_emb if img_emb is not None else fused

    def set_backbone_trainable(self, flag: bool) -> None:
        if self.use_image:
            for p in self.backbone.parameters():
                p.requires_grad = flag

    def parameter_counts(self) -> Dict[str, int]:
        out = {"total": sum(p.numel() for p in self.parameters())}
        if self.use_image:
            out["backbone"] = sum(p.numel() for p in self.backbone.parameters())
            out["vit"] = sum(p.numel() for p in self.vit.parameters())
        if self.use_pnf:
            out["numeric_mlp"] = sum(p.numel() for p in self.numeric_mlp.parameters())
        out["head"] = sum(p.numel() for p in self.head.parameters())
        return out


def angular_aware_loss(logits: torch.Tensor, labels: torch.Tensor, bins: BinScheme,
                       class_weights: Optional[torch.Tensor], lambda_angular: float,
                       label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Weighted cross-entropy (optionally label-smoothed) plus an expected
    circular-distance penalty between the predicted bin distribution and the
    true bin. The penalty is normalised so that 180 degrees maps to 1.
    """
    ce = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
    if lambda_angular <= 0:
        return ce
    centres = torch.tensor(bins.centres(), dtype=torch.float32, device=logits.device)
    diff = torch.abs(centres.unsqueeze(0) - centres[labels].unsqueeze(1))
    dist = torch.minimum(diff, 360.0 - diff) / 180.0
    probs = torch.softmax(logits, dim=1)
    return ce + lambda_angular * (probs * dist).sum(dim=1).mean()


# ----------------------------------------------------------------------------
# Phase 2 regressors
# ----------------------------------------------------------------------------

class BinRegressor:
    """
    Regressor for one angular bin.

    target_mode:
      'cossin'  : two XGBoost models predicting cos(roll) and sin(roll);
                  the angle is recovered with atan2(sin, cos).
      'degrees' : one XGBoost model predicting the roll angle directly.
    """

    def __init__(self, target_mode: str, params: Dict, seed: int = 42):
        assert target_mode in ("cossin", "degrees")
        if xgb is None:
            raise ImportError("xgboost is required for Phase 2 regressors")
        self.target_mode = target_mode
        n_out = 2 if target_mode == "cossin" else 1
        self.models = [xgb.XGBRegressor(**params, random_state=seed + k) for k in range(n_out)]
        self.n_train = 0

    def fit(self, X: np.ndarray, roll_deg: np.ndarray) -> "BinRegressor":
        self.n_train = len(roll_deg)
        if self.target_mode == "cossin":
            y = angle_to_cossin(roll_deg)
            self.models[0].fit(X, y[:, 0])
            self.models[1].fit(X, y[:, 1])
        else:
            self.models[0].fit(X, roll_deg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.target_mode == "cossin":
            tx = self.models[0].predict(X)
            ty = self.models[1].predict(X)
            return cossin_to_angle(tx, ty)
        return wrap_deg(self.models[0].predict(X))


def train_bin_regressors(features: np.ndarray, roll: np.ndarray, gt_bins: np.ndarray,
                         n_bins: int, target_mode: str, params: Dict, seed: int,
                         min_samples: int = 2, neighbor_overlap: int = 0) -> List[Optional[BinRegressor]]:
    """
    Train one BinRegressor per ground-truth bin. With neighbor_overlap = k, the
    regressor of bin i is trained on samples from bins i-k ... i+k (circular),
    so that a sample routed to an adjacent bin is still handled by a regressor
    that has seen its angular range. Bins with too few samples get None.
    """
    regs: List[Optional[BinRegressor]] = []
    for i in range(n_bins):
        members = [(i + d) % n_bins for d in range(-neighbor_overlap, neighbor_overlap + 1)]
        m = np.isin(gt_bins, members)
        if m.sum() < min_samples:
            regs.append(None)
            continue
        regs.append(BinRegressor(target_mode, params, seed).fit(features[m], roll[m]))
    return regs


def route_and_predict(probs: np.ndarray, features: np.ndarray,
                      regressors: Sequence[Optional[BinRegressor]],
                      bins: BinScheme, soft_top_k: int = 1) -> Dict[str, np.ndarray]:
    """
    Hard routing (soft_top_k = 1): each sample goes to the regressor of its
    argmax bin (falling back to the next most probable bin that has a
    regressor). Boundary clipping replaces a prediction lying outside the
    routed bin (shortest-arc distance to the bin centre) by the bin centre.

    Soft routing (soft_top_k > 1): the outputs of the top-k regressors are
    averaged as unit vectors, weighted by their calibrated probabilities. The
    result is not clipped. 'disagreement' is the circular standard deviation
    (deg) of the top-k regressor outputs and can serve as an uncertainty cue.
    """
    n = len(features)
    raw = np.zeros(n)
    clipped = np.zeros(n)
    routed_bin = np.zeros(n, dtype=int)
    outcome = np.empty(n, dtype=object)
    disagreement = np.zeros(n)
    order = np.argsort(-probs, axis=1)
    if soft_top_k > 1:
        for i in range(n):
            cand = [int(b) for b in order[i] if regressors[b] is not None][:soft_top_k]
            if not cand:
                raw[i] = clipped[i] = bins.centre(int(order[i][0])); routed_bin[i] = int(order[i][0])
                outcome[i] = "no_regressor"; continue
            angs = np.array([float(regressors[b].predict(features[i:i + 1])[0]) for b in cand])
            w = probs[i, cand]; w = w / max(w.sum(), 1e-12)
            uv = angle_to_cossin(angs)
            mean_vec = (uv * w[:, None]).sum(axis=0)
            ang = float(cossin_to_angle(mean_vec[0], mean_vec[1]))
            raw[i] = clipped[i] = ang; routed_bin[i] = cand[0]; outcome[i] = "soft"
            r = float(np.clip(np.sqrt((mean_vec ** 2).sum()), 1e-12, 1.0))
            disagreement[i] = float(np.rad2deg(np.sqrt(-2.0 * np.log(r))))
        return {"raw": raw, "clipped": clipped, "routed_bin": routed_bin, "outcome": outcome,
                "disagreement": disagreement}
    for i in range(n):
        chosen = None
        for b in order[i]:
            if regressors[b] is not None:
                chosen = int(b)
                break
        if chosen is None:
            raw[i] = clipped[i] = bins.centre(int(order[i][0]))
            routed_bin[i] = int(order[i][0])
            outcome[i] = "no_regressor"
            continue
        pred = float(regressors[chosen].predict(features[i:i + 1])[0])
        raw[i] = pred
        routed_bin[i] = chosen
        if bins.contains(pred, chosen):
            clipped[i] = pred
            outcome[i] = "normal" if chosen == order[i][0] else "fallback"
        else:
            clipped[i] = bins.centre(chosen)
            outcome[i] = "clipped"
        # disagreement between the top-2 regressors (0 if only one is available)
        second = [int(b) for b in order[i] if regressors[b] is not None and int(b) != chosen][:1]
        if second:
            a2 = float(regressors[second[0]].predict(features[i:i + 1])[0])
            disagreement[i] = circular_error_deg(np.array([pred]), np.array([a2]))[0]
    return {"raw": raw, "clipped": clipped, "routed_bin": routed_bin, "outcome": outcome,
            "disagreement": disagreement}


# ----------------------------------------------------------------------------
# Confidence Estimation Pipeline (CEP)
# ----------------------------------------------------------------------------

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single-parameter temperature minimising validation NLL."""
    def nll(t):
        p = softmax_np(logits / max(t, 1e-3))
        return -np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)))
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def uncertainty_features(probs: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per-sample [p_max, p_second, margin, entropy] from a (n, N) probability
    matrix, optionally followed by extra uncertainty cues (e.g. regressor
    disagreement) given as an (n, m) array.
    """
    s = -np.sort(-probs, axis=1)
    p1 = s[:, 0]
    p2 = s[:, 1] if probs.shape[1] > 1 else np.zeros_like(p1)
    ent = -np.sum(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1)), axis=1)
    f = np.stack([p1, p2, p1 - p2, ent], axis=1)
    if extra is not None:
        f = np.concatenate([f, np.asarray(extra, dtype=np.float64).reshape(len(f), -1)], axis=1)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class CEP:
    """
    Confidence Estimation Pipeline fitted on the validation split:
      temperature -> meta-classifier P(Phase 1 correct) -> acceptance threshold.
    """
    temperature: float
    scaler: StandardScaler
    meta: LogisticRegression
    threshold: float
    threshold_found: bool
    target_mae_deg: float
    min_coverage_pct: float
    val_mae_conf: float = float("nan")
    val_coverage_pct: float = float("nan")
    use_disagreement: bool = False

    def calibrated_probs(self, logits: np.ndarray) -> np.ndarray:
        return softmax_np(logits / self.temperature)

    def p_correct(self, probs: np.ndarray, disagreement: Optional[np.ndarray] = None) -> np.ndarray:
        extra = disagreement if self.use_disagreement else None
        return self.meta.predict_proba(self.scaler.transform(uncertainty_features(probs, extra)))[:, 1]

    def to_json(self) -> Dict:
        return {
            "temperature": self.temperature,
            "threshold": self.threshold,
            "threshold_found": bool(self.threshold_found),
            "target_mae_deg": self.target_mae_deg,
            "min_coverage_pct": self.min_coverage_pct,
            "val_mae_conf": self.val_mae_conf,
            "val_coverage_pct": self.val_coverage_pct,
            "use_disagreement": bool(self.use_disagreement),
            "meta_coef": self.meta.coef_.ravel().tolist(),
            "meta_intercept": float(self.meta.intercept_[0]),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }


def fit_cep(val_logits: np.ndarray, val_gt_bins: np.ndarray, val_pred_bins: np.ndarray,
            val_pred_roll_clipped: np.ndarray, val_true_roll: np.ndarray,
            target_mae_deg: float, min_coverage_pct: float,
            thr_grid: Sequence[float], fallback_threshold: float = 0.5,
            seed: int = 42, val_disagreement: Optional[np.ndarray] = None) -> Tuple[CEP, pd.DataFrame]:
    """
    Fit all CEP components on validation data and return (CEP, threshold sweep
    table). If val_disagreement is given, the regressor-disagreement cue is
    added to the meta-classifier inputs.
    """
    T = fit_temperature(val_logits, val_gt_bins)
    probs = softmax_np(val_logits / T)
    X = uncertainty_features(probs, val_disagreement)
    y = (val_pred_bins == val_gt_bins).astype(int)

    scaler = StandardScaler().fit(X)
    if y.min() == y.max():
        # Degenerate validation set (all correct or all wrong): constant meta-classifier
        meta = LogisticRegression(max_iter=1000, random_state=seed)
        meta.fit(np.vstack([scaler.transform(X), scaler.transform(X)]),
                 np.concatenate([y, 1 - y]))
    else:
        meta = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
        meta.fit(scaler.transform(X), y)
    p_corr = meta.predict_proba(scaler.transform(X))[:, 1]

    err = circular_error_deg(val_pred_roll_clipped, val_true_roll)
    rows = []
    chosen, found = None, False
    for thr in thr_grid:
        m = p_corr >= thr
        cov = 100.0 * m.mean()
        mae = float(err[m].mean()) if m.any() else float("nan")
        rows.append({"threshold": round(float(thr), 4), "val_mae_conf": mae, "val_coverage_pct": cov,
                     "n_conf": int(m.sum())})
        if chosen is None and m.any() and mae <= target_mae_deg and cov >= min_coverage_pct:
            chosen, found = float(thr), True
    sweep = pd.DataFrame(rows)
    if chosen is None:
        valid = sweep.dropna(subset=["val_mae_conf"])
        chosen = float(valid.loc[valid["val_mae_conf"].idxmin(), "threshold"]) if len(valid) else fallback_threshold
    row = sweep[np.isclose(sweep["threshold"], chosen)].iloc[0] if (np.isclose(sweep["threshold"], chosen)).any() else None
    cep = CEP(temperature=T, scaler=scaler, meta=meta, threshold=chosen, threshold_found=found,
              target_mae_deg=target_mae_deg, min_coverage_pct=min_coverage_pct,
              val_mae_conf=float(row["val_mae_conf"]) if row is not None else float("nan"),
              val_coverage_pct=float(row["val_coverage_pct"]) if row is not None else float("nan"),
              use_disagreement=val_disagreement is not None)
    return cep, sweep


# ----------------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------------

def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def risk_coverage_curve(err: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """
    Selective risk (mean circular error of accepted samples) as a function of
    coverage, accepting samples in decreasing order of confidence `score`.
    """
    order = np.argsort(-score)
    e = err[order]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    return pd.DataFrame({"coverage": cov, "risk_mae": cum, "score_threshold": score[order]})


def area_under_risk_coverage(rc: pd.DataFrame) -> float:
    """AURC: integral of selective risk over coverage in [0, 1] (trapezoidal)."""
    cov = np.concatenate([[0.0], rc["coverage"].values])
    risk = np.concatenate([[rc["risk_mae"].values[0]], rc["risk_mae"].values])
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(risk, cov))


def binary_metrics(score: np.ndarray, positive: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Treats `positive` (e.g. error <= target) as the event to detect and `score`
    (P(correct)) as the detector. Returns AUROC, AUPRC and operating-point
    sensitivity/specificity/false-negative rate for large errors.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = {}
    pos = positive.astype(int)
    if pos.min() != pos.max():
        out["auroc"] = float(roc_auc_score(pos, score))
        out["auprc"] = float(average_precision_score(pos, score))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    acc = score >= threshold
    tp = np.sum(acc & (pos == 1)); fn = np.sum(~acc & (pos == 1))
    tn = np.sum(~acc & (pos == 0)); fp = np.sum(acc & (pos == 0))
    out["sensitivity_small_error_accepted"] = float(tp / max(tp + fn, 1))
    out["specificity_large_error_rejected"] = float(tn / max(tn + fp, 1))
    out["large_error_accept_rate"] = float(fp / max(tn + fp, 1))  # false-negative rate of the safety filter
    out["n_large_error_accepted"] = int(fp)
    return out


def summarize_predictions(true_roll: np.ndarray, raw: np.ndarray, clipped: np.ndarray,
                          p_correct: np.ndarray, threshold: float, target_mae_deg: float,
                          gt_bins: np.ndarray, pred_bins: np.ndarray, probs_cal: np.ndarray,
                          n_bins: int) -> Dict[str, float]:
    """All scalar metrics reported for one split."""
    err_raw = circular_error_deg(raw, true_roll)
    err_clip = circular_error_deg(clipped, true_roll)
    conf = p_correct >= threshold
    n = len(true_roll)
    out = {
        "n": n,
        "mae1_all_raw": float(err_raw.mean()),
        "mae2_all_clipped": float(err_clip.mean()),
        "std2_all_clipped": float(err_clip.std()),
        "median_err_all": float(np.median(err_clip)),
        "mae3_confident": float(err_clip[conf].mean()) if conf.any() else float("nan"),
        "std3_confident": float(err_clip[conf].std()) if conf.any() else float("nan"),
        "mae_uncertain": float(err_clip[~conf].mean()) if (~conf).any() else float("nan"),
        "n_confident": int(conf.sum()),
        "n_uncertain": int((~conf).sum()),
        "coverage_pct": 100.0 * conf.mean(),
        "pct_within_target_all": 100.0 * float((err_clip <= target_mae_deg).mean()),
        "pct_within_target_confident": 100.0 * float((err_clip[conf] <= target_mae_deg).mean()) if conf.any() else float("nan"),
        "r2_circular": circular_r2(clipped, true_roll),
        "circular_correlation": circular_correlation(clipped, true_roll),
        "phase1_accuracy": float((gt_bins == pred_bins).mean()),
        "phase1_adjacent_accuracy": float((np.minimum(np.abs(gt_bins - pred_bins),
                                                       n_bins - np.abs(gt_bins - pred_bins)) <= 1).mean()),
        "ece_phase1": expected_calibration_error(probs_cal.max(axis=1), (gt_bins == pred_bins).astype(float)),
        "nll_phase1": float(-np.mean(np.log(np.clip(probs_cal[np.arange(n), gt_bins], 1e-12, 1)))),
        "brier_meta": float(np.mean((p_correct - (gt_bins == pred_bins)) ** 2)),
        "ece_meta": expected_calibration_error(p_correct, (gt_bins == pred_bins).astype(float)),
    }
    rc = risk_coverage_curve(err_clip, p_correct)
    out["aurc"] = area_under_risk_coverage(rc)
    # Oracle ordering (smallest true error first) gives the lowest AURC achievable
    # for these predictions; the excess over it isolates the quality of the confidence score.
    out["aurc_oracle"] = area_under_risk_coverage(risk_coverage_curve(err_clip, -err_clip))
    out["aurc_excess"] = out["aurc"] - out["aurc_oracle"]
    out.update({f"filter_{k}": v for k, v in
                binary_metrics(p_correct, err_clip <= target_mae_deg, threshold).items()})
    return out


# ----------------------------------------------------------------------------
# Temporal post-processing within consecutive frame blocks
# ----------------------------------------------------------------------------

def _causal_circular_mean(prev: np.ndarray) -> float:
    uv = angle_to_cossin(prev).mean(axis=0)
    return float(cossin_to_angle(uv[0], uv[1]))


def temporal_postprocess(pred: np.ndarray, confident: np.ndarray, block_id: np.ndarray,
                         frame_index: np.ndarray, window: int) -> Dict[str, np.ndarray]:
    """
    Causal post-processing inside each block of consecutive frames (ordered by
    frame_index). Only past and current frames are used, so the procedure is
    deployable in real time.

      smoothed  : circular mean of the predictions over the last `window` frames
      fallback  : hold-last-confident policy - an uncertain frame takes the most
                  recent confident prediction in the same block; if none exists
                  yet, its own prediction is kept and the frame is marked
                  'unfilled'
      smoothed_fallback : fallback output followed by the same causal smoothing
    """
    n = len(pred)
    smoothed = pred.astype(np.float64).copy()
    fallback = pred.astype(np.float64).copy()
    filled = np.zeros(n, dtype=bool)
    sm_fb = pred.astype(np.float64).copy()
    df = pd.DataFrame({"i": np.arange(n), "b": block_id, "f": frame_index})
    for _, g in df.groupby("b", sort=False):
        order = g.sort_values("f")["i"].values
        last_conf = None
        for k, i in enumerate(order):
            lo = max(0, k - window + 1)
            smoothed[i] = _causal_circular_mean(pred[order[lo:k + 1]])
            if confident[i]:
                last_conf = pred[i]
            elif last_conf is not None:
                fallback[i] = last_conf; filled[i] = True
            sm_fb[i] = _causal_circular_mean(fallback[order[lo:k + 1]])
    return {"smoothed": smoothed, "fallback": fallback, "fallback_filled": filled, "smoothed_fallback": sm_fb}


def uncertain_run_lengths(confident: np.ndarray, block_id: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    """Lengths (in frames) of consecutive uncertain runs inside blocks."""
    runs = []
    df = pd.DataFrame({"c": confident, "b": block_id, "f": frame_index})
    for _, g in df.groupby("b", sort=False):
        c = g.sort_values("f")["c"].values
        k = 0
        for v in c:
            if not v:
                k += 1
            elif k:
                runs.append(k); k = 0
        if k:
            runs.append(k)
    return np.array(runs, dtype=int)


# ----------------------------------------------------------------------------
# Persistence and end-to-end predictor
# ----------------------------------------------------------------------------

def save_model_bundle(out_dir: str, model: HybridRViT, regressors: Sequence[Optional[BinRegressor]],
                      cep: CEP, normalizer: NumericNormalizer, config: Dict) -> None:
    """Write everything needed for inference into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "phase1_classifier.pth"))
    for i, r in enumerate(regressors):
        if r is not None:
            joblib.dump(r, os.path.join(out_dir, f"xgb_bin_{i:02d}.pkl"))
    joblib.dump({"scaler": cep.scaler, "meta": cep.meta}, os.path.join(out_dir, "cep_meta.pkl"))
    with open(os.path.join(out_dir, "cep.json"), "w") as f:
        json.dump(cep.to_json(), f, indent=2)
    bundle_cfg = dict(config)
    bundle_cfg["numeric_mean"] = normalizer.mean
    bundle_cfg["numeric_std"] = normalizer.std
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(bundle_cfg, f, indent=2)


class DACRViTPredictor:
    """
    End-to-end inference for a saved model bundle.

    predict(image_paths, pnf) -> dict with keys:
        roll_deg, roll_raw_deg, bin, p_correct, is_confident, probs
    """

    def __init__(self, bundle_dir: str, device: Optional[str] = None):
        with open(os.path.join(bundle_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        n = int(self.cfg["n_bins"])
        self.bins = BinScheme(n)
        self.model = HybridRViT(num_classes=n, embed_dim=self.cfg["embed_dim"],
                                num_heads=self.cfg["num_heads"], num_layers=self.cfg["num_layers"],
                                ff_dim=self.cfg["ff_dim"], dropout=self.cfg["dropout"],
                                image_size=self.cfg["image_size"], input_mode=self.cfg["input_mode"],
                                pretrained_backbone=False)
        state = torch.load(os.path.join(bundle_dir, "phase1_classifier.pth"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.regressors: List[Optional[BinRegressor]] = []
        for i in range(n):
            p = os.path.join(bundle_dir, f"xgb_bin_{i:02d}.pkl")
            self.regressors.append(joblib.load(p) if os.path.exists(p) else None)
        with open(os.path.join(bundle_dir, "cep.json")) as f:
            cj = json.load(f)
        meta = joblib.load(os.path.join(bundle_dir, "cep_meta.pkl"))
        self.cep = CEP(temperature=cj["temperature"], scaler=meta["scaler"], meta=meta["meta"],
                       threshold=cj["threshold"], threshold_found=cj["threshold_found"],
                       target_mae_deg=cj["target_mae_deg"], min_coverage_pct=cj["min_coverage_pct"],
                       use_disagreement=bool(cj.get("use_disagreement", False)))
        self.normalizer = NumericNormalizer(self.cfg["numeric_mean"], self.cfg["numeric_std"])

    @torch.no_grad()
    def predict(self, image_paths: Sequence[str], pnf: np.ndarray, batch_size: int = 32) -> Dict[str, np.ndarray]:
        pnf_n = torch.from_numpy(self.normalizer(np.asarray(pnf, dtype=np.float32)))
        logits_all, emb_all = [], []
        for s in range(0, len(image_paths), batch_size):
            paths = image_paths[s:s + batch_size]
            x = None
            if self.model.use_image:
                x = torch.stack([load_image_tensor(p, self.cfg["image_size"]) for p in paths])
                x = normalize_batch(x.to(self.device))
            q = pnf_n[s:s + batch_size].to(self.device) if self.model.use_pnf else None
            logits_all.append(self.model(x, q).cpu().numpy())
            emb_all.append(self.model.embedding(x, q).cpu().numpy())
        logits = np.concatenate(logits_all)
        emb = np.concatenate(emb_all)
        feats = np.concatenate([emb, pnf_n.numpy()], axis=1) if self.model.use_pnf else emb
        probs = self.cep.calibrated_probs(logits)
        routed = route_and_predict(probs, feats, self.regressors, self.bins,
                                   soft_top_k=int(self.cfg.get("soft_top_k", 1)))
        p_corr = self.cep.p_correct(probs, routed["disagreement"])
        return {"roll_deg": routed["clipped"], "roll_raw_deg": routed["raw"],
                "bin": routed["routed_bin"], "clip_outcome": routed["outcome"],
                "disagreement": routed["disagreement"],
                "p_correct": p_corr, "is_confident": p_corr >= self.cep.threshold, "probs": probs}
