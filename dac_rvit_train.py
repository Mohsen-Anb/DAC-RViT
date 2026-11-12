"""
DAC-RViT training and evaluation pipeline.

Stages
  A. N sweep        : one seed per N; the number of bins is selected on the
                      validation split only.
  B. Seed stability : several seeds at the selected N; the released checkpoint
                      is the seed with the best validation performance; all
                      seeds are reported on the test split.
  C. Controls       : image-only and PNF-only classifiers; shuffled-PNF test.
  D. Baselines      : global regressors and bin-centre baselines on the same
                      splits and the same circular metric.

Every run writes a self-contained folder with the model bundle, the split
manifest, per-frame predictions, metrics and figures. Summary tables and
figures are written to CONFIG['results_root'].

Run `python dac_rvit_train.py` after editing CONFIG below. Set
CONFIG['smoke_test'] = True for a quick end-to-end check on a small fraction
of the data.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dac_rvit_core import (  # noqa: E402
    BinScheme, CEP, HybridRViT, NumericNormalizer, BinRegressor,
    angle_to_cossin, angular_aware_loss, area_under_risk_coverage,
    boundary_adjacency_stats, build_manifest, circular_error_deg, cossin_to_angle,
    fit_cep, load_image_tensor, normalize_batch, photometric_augment,
    risk_coverage_curve, route_and_predict, save_model_bundle, seed_everything,
    softmax_np, summarize_predictions, temporal_postprocess, train_bin_regressors,
    uncertain_run_lengths, wrap_deg,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG: Dict = {
    # ---- data -----------------------------------------------------------
    "main_folder": "data/Extracted_Frames_Roll_X_Y_4_ViT",
    "results_root": "Results_full_frame1",
    "data_fraction": 1.0,          # fraction of blocks used (per recording); 0.02 for a quick check
    "block_len": 1,               # consecutive frames per split block (15 frames = 1 s at 15 fps)
    "train_frac": 0.70,
    "val_frac": 0.15,              # test fraction is the remainder
    "image_size": 224,

 # ---- Phase 1 (classifier) ------------------------------------------
    "embed_dim": 300,
    "num_heads": 10,
    "num_layers": 4,
    "ff_dim": 1200,
    "dropout": 0.1,
    "batch_size": 8,
    "epochs": 250,
    "warmup_epochs": 5,            # backbone frozen during warm-up
    "lr": 3e-4,
    "lr_after_unfreeze": 3e-5,
    "weight_decay": 0.01,
    "patience": 20,                # early stopping on validation accuracy
    "label_smoothing": 0.0,
    "target_val_acc": 0.95,        # stop early when reached
    "angular_lambda": 0.5,         # weight of the angular penalty in the Phase 1 loss
    "class_weighting": True,
    "grad_clip": 1.0,
    "num_workers": 4,
    "pretrained_backbone": True,
    "augment": {                    # photometric only; applied to the training split
        "brightness": 0.2,
        "contrast": 0.2,
        "gaussian_noise_std": 0.01,
        "blur_prob": 0.2,
        "blur_kernel": 3,
        "blur_sigma": 0.8,
        "erase_prob": 0.0,          # random erasing of one rectangle per image
    },

    # ---- Phase 2 (regressors) ------------------------------------------
    "target_mode": "cossin",       # primary: 'cossin'; 'degrees' is also evaluated for comparison
    "compare_target_modes": True,
    "xgb_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                   "n_jobs": 4, "tree_method": "hist"},
    "min_bin_samples": 2,
    "neighbor_overlap": 1,         # k: regressor i trains on bins i-k..i+k (0 = own bin only)
    "soft_top_k": 3,               # 1 = hard routing with clipping; >1 = probability-weighted top-k average

    # ---- Confidence Estimation Pipeline --------------------------------
    "target_mae_deg": 10.0,        # clinically justified acceptance target (confident subset)
    "min_coverage_pct": 50.0,
    "cep_use_disagreement": True, # add top-2 regressor disagreement to the meta-classifier inputs
    "temporal_window": 5,          # causal smoothing window (frames) inside consecutive blocks; 0 = off
    "threshold_grid": [round(x, 2) for x in np.arange(0.30, 1.00, 0.01)],

    # ---- experiment design -----------------------------------------------
    #"n_values": [14],
    "n_values": [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 40],
    "sweep_seed": 42,
    #"stability_seeds": [42],
    "stability_seeds": [42, 43, 44, 45, 46],
    "n_selection": "composite",    # 'composite': min MAE_all * MAE_conf * AURC * (100 - coverage) on validation
                                   # 'mae_conf' : min validation confident-subset MAE
    "reuse_existing_runs": True,   # a run folder that already contains metrics.csv is loaded instead of retrained
    "run_controls": True,          # image-only and PNF-only classifiers + shuffled-PNF test
    "run_baselines": True,         # global regressors and bin-centre baselines
    "sequence_plot_len": 200,      # number of test frames shown in the prediction trace figure
    "deterministic": True,

    # ---- quick end-to-end check -----------------------------------------
    "smoke_test": False,           # True -> tiny data fraction, few epochs, N in [4], one seed
    "smoke_overrides": {"data_fraction": 0.05, "epochs": 2, "warmup_epochs": 1, "patience": 5,
                        "n_values": [4], "stability_seeds": [42, 43], "num_workers": 0,
                        "sequence_plot_len": 20, "results_root": "results_smoke"},
}


# ============================================================================
# Data handling
# ============================================================================

class FrameCache:
    """Loads every frame once into a uint8 tensor cache and serves batches by index."""

    def __init__(self, manifest: pd.DataFrame, image_size: int, use_image: bool):
        self.manifest = manifest.reset_index(drop=True)
        self.use_image = use_image
        self.images = None
        if use_image:
            n = len(self.manifest)
            self.images = torch.empty((n, 3, image_size, image_size), dtype=torch.uint8)
            for i, p in enumerate(self.manifest["path"].values):
                self.images[i] = load_image_tensor(p, image_size)
        self.roll = self.manifest["roll"].values.astype(np.float64)
        self.pnf_raw = self.manifest[["pnf_0", "pnf_1", "pnf_2", "pnf_3"]].values.astype(np.float32)

    def indices(self, split: str) -> np.ndarray:
        return np.where(self.manifest["split"].values == split)[0]


def iterate_batches(idx: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.RandomState):
    idx = idx.copy()
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield idx[s:s + batch_size]


def make_inputs(cache: FrameCache, idx: np.ndarray, pnf_norm: np.ndarray, device: torch.device,
                augment: Optional[Dict], use_image: bool, use_pnf: bool):
    x = None
    if use_image:
        x = cache.images[idx].to(device)
        if augment:
            x = photometric_augment(x, augment)
        x = normalize_batch(x)
    q = torch.from_numpy(pnf_norm[idx]).to(device) if use_pnf else None
    return x, q


# ============================================================================
# Phase 1 training
# ============================================================================

def train_phase1(cache: FrameCache, pnf_norm: np.ndarray, bins: BinScheme, cfg: Dict,
                 device: torch.device, seed: int, input_mode: str, out_dir: str) -> Tuple[HybridRViT, pd.DataFrame, Dict]:
    """Train the Phase 1 classifier and return (best model, learning curve table, info)."""
    seed_everything(seed, cfg["deterministic"])
    rng = np.random.RandomState(seed)
    model = HybridRViT(num_classes=bins.n_bins, embed_dim=cfg["embed_dim"], num_heads=cfg["num_heads"],
                       num_layers=cfg["num_layers"], ff_dim=cfg["ff_dim"], dropout=cfg["dropout"],
                       image_size=cfg["image_size"], input_mode=input_mode,
                       pretrained_backbone=cfg["pretrained_backbone"]).to(device)
    use_image, use_pnf = model.use_image, model.use_pnf

    tr_idx, va_idx = cache.indices("train"), cache.indices("val")
    gt = bins.index(cache.roll)
    counts = np.bincount(gt[tr_idx], minlength=bins.n_bins).astype(np.float64)
    class_w = None
    if cfg["class_weighting"]:
        w = np.where(counts > 0, counts.sum() / (bins.n_bins * np.maximum(counts, 1)), 1.0)
        class_w = torch.tensor(w, dtype=torch.float32, device=device)

    model.set_backbone_trainable(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0,
                                             total_iters=max(cfg["warmup_epochs"], 1))
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    rows, best_acc, best_epoch, best_state, bad = [], -1.0, 0, None, 0
    for epoch in range(cfg["epochs"]):
        if epoch == cfg["warmup_epochs"] and use_image:
            model.set_backbone_trainable(True)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr_after_unfreeze"],
                                   weight_decay=cfg["weight_decay"])
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

        model.train()
        tl, tc, tn = 0.0, 0, 0
        for b in iterate_batches(tr_idx, cfg["batch_size"], True, rng):
            x, q = make_inputs(cache, b, pnf_norm, device, cfg["augment"], use_image, use_pnf)
            y = torch.from_numpy(gt[b]).to(device)
            opt.zero_grad()
            logits = model(x, q)
            loss = angular_aware_loss(logits, y, bins, class_w, cfg["angular_lambda"], cfg.get("label_smoothing", 0.0))
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tl += loss.item() * len(b); tc += (logits.argmax(1) == y).sum().item(); tn += len(b)

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for b in iterate_batches(va_idx, cfg["batch_size"], False, rng):
                x, q = make_inputs(cache, b, pnf_norm, device, None, use_image, use_pnf)
                y = torch.from_numpy(gt[b]).to(device)
                logits = model(x, q)
                vl += angular_aware_loss(logits, y, bins, class_w, cfg["angular_lambda"], cfg.get("label_smoothing", 0.0)).item() * len(b)
                vc += (logits.argmax(1) == y).sum().item(); vn += len(b)
        tr_loss, tr_acc, va_loss, va_acc = tl / max(tn, 1), tc / max(tn, 1), vl / max(vn, 1), vc / max(vn, 1)
        rows.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc, "lr": opt.param_groups[0]["lr"]})
        print(f"    epoch {epoch + 1:3d}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if epoch < cfg["warmup_epochs"]:
            warm.step()
        else:
            plateau.step(va_loss)

        if va_acc > best_acc:
            best_acc, best_epoch, bad = va_acc, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"] or va_acc >= cfg["target_val_acc"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(out_dir, "learning_curves.csv"), index=False)
    return model, curves, {"best_val_acc": best_acc, "best_epoch": best_epoch,
                           "epochs_run": len(rows), "params": model.parameter_counts()}


@torch.no_grad()
def extract(cache: FrameCache, pnf_norm: np.ndarray, model: HybridRViT, idx: np.ndarray,
            device: torch.device, batch_size: int, pnf_override: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Frozen forward pass: logits and Phase 2 features for the given frame indices."""
    pn = pnf_override if pnf_override is not None else pnf_norm
    logits, emb = [], []
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        x, q = make_inputs(cache, b, pn, device, None, model.use_image, model.use_pnf)
        logits.append(model(x, q).cpu().numpy())
        emb.append(model.embedding(x, q).cpu().numpy())
    logits = np.concatenate(logits); emb = np.concatenate(emb)
    feats = np.concatenate([emb, pn[idx]], axis=1) if model.use_pnf else emb
    return {"logits": logits, "features": feats, "roll": cache.roll[idx], "idx": idx}


# ============================================================================
# One complete run (Phase 1 + Phase 2 + CEP + evaluation)
# ============================================================================

def load_existing_run(out_dir: str) -> Optional[Dict]:
    """Load the metrics row (and any baselines/controls) of a completed run folder."""
    mp = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(mp):
        return None
    m = pd.read_csv(mp).iloc[0].to_dict()
    bp = os.path.join(out_dir, "baselines_and_controls.csv")
    if os.path.exists(bp):
        b = pd.read_csv(bp)
        m["extras"] = {r["baseline"]: {k: v for k, v in r.items() if k != "baseline"} for _, r in b.iterrows()}
    return m


def run_experiment(n_bins: int, seed: int, cfg: Dict, device: torch.device, out_dir: str,
                   input_mode: str = "both", tag: str = "") -> Dict:
    if cfg.get("reuse_existing_runs", True):
        existing = load_existing_run(out_dir)
        if existing is not None:
            print(f"\n=== Reusing completed run: {out_dir}")
            return existing
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    bins = BinScheme(n_bins)
    print(f"\n=== Run: N={n_bins} seed={seed} mode={input_mode} {tag} -> {out_dir}")

    manifest = build_manifest(cfg["main_folder"], cfg["block_len"], cfg["train_frac"], cfg["val_frac"],
                              seed, cfg["data_fraction"])
    manifest.to_csv(os.path.join(out_dir, "split_manifest.csv"), index=False)
    adj = boundary_adjacency_stats(manifest)
    cache = FrameCache(manifest, cfg["image_size"], use_image=(input_mode != "pnf"))
    tr, va, te = cache.indices("train"), cache.indices("val"), cache.indices("test")
    normalizer = NumericNormalizer.fit(cache.pnf_raw[tr])
    pnf_norm = normalizer(cache.pnf_raw)
    print(f"    frames: train {len(tr)}  val {len(va)}  test {len(te)}  "
          f"(test frames adjacent to train: {adj['pct_test_adjacent_to_train']:.1f}%)")

    # ---- Phase 1 -----------------------------------------------------------
    model, curves, p1 = train_phase1(cache, pnf_norm, bins, cfg, device, seed, input_mode, out_dir)
    plot_learning_curves(curves, os.path.join(out_dir, "fig_learning_curves"))

    S = {name: extract(cache, pnf_norm, model, idx, device, 32) for name, idx in
         (("train", tr), ("val", va), ("test", te))}
    for s in S.values():
        s["gt_bin"] = bins.index(s["roll"])
        s["pred_bin"] = s["logits"].argmax(1)

    # ---- Phase 2 (both target modes when requested) -------------------------
    modes = [cfg["target_mode"]] + ([m for m in ("cossin", "degrees") if m != cfg["target_mode"]]
                                     if cfg["compare_target_modes"] else [])
    per_mode: Dict[str, Dict] = {}
    for mode in modes:
        regs = train_bin_regressors(S["train"]["features"], S["train"]["roll"], S["train"]["gt_bin"],
                                    n_bins, mode, cfg["xgb_params"], seed, cfg["min_bin_samples"],
                                    neighbor_overlap=cfg.get("neighbor_overlap", 0))
        k_soft = int(cfg.get("soft_top_k", 1))
        # Route validation samples through the regressors before fitting the CEP
        # (temperature is fitted inside fit_cep; routing here uses the raw softmax).
        va_probs0 = softmax_np(S["val"]["logits"])
        va_route0 = route_and_predict(va_probs0, S["val"]["features"], regs, bins, soft_top_k=k_soft)
        cep, sweep = fit_cep(S["val"]["logits"], S["val"]["gt_bin"], S["val"]["pred_bin"],
                             va_route0["clipped"], S["val"]["roll"], cfg["target_mae_deg"],
                             cfg["min_coverage_pct"], cfg["threshold_grid"], seed=seed,
                             val_disagreement=va_route0["disagreement"] if cfg.get("cep_use_disagreement") else None)
        res = {}
        for name, s in S.items():
            probs = cep.calibrated_probs(s["logits"])
            route = route_and_predict(probs, s["features"], regs, bins, soft_top_k=k_soft)
            pc = cep.p_correct(probs, route["disagreement"])
            res[name] = {"probs": probs, "route": route, "p_correct": pc,
                         "_err": circular_error_deg(route["clipped"], s["roll"]), "_thr": cep.threshold,
                         "metrics": summarize_predictions(s["roll"], route["raw"], route["clipped"], pc,
                                                          cep.threshold, cfg["target_mae_deg"],
                                                          s["gt_bin"], probs.argmax(1), probs, n_bins)}
        per_mode[mode] = {"regs": regs, "cep": cep, "sweep": sweep, "res": res}

    primary = per_mode[cfg["target_mode"]]
    cep, regs, res = primary["cep"], primary["regs"], primary["res"]
    primary["sweep"].to_csv(os.path.join(out_dir, "threshold_sweep_val.csv"), index=False)

    # ---- per-frame predictions and per-bin table ---------------------------
    for name, s in S.items():
        r = res[name]["route"]
        df = manifest.iloc[s["idx"]][["recording", "frame_index", "block_id", "split"]].copy()
        df["true_roll"] = s["roll"]; df["pred_roll"] = r["clipped"]; df["pred_roll_raw"] = r["raw"]
        df["error_circular"] = circular_error_deg(r["clipped"], s["roll"])
        df["gt_bin"] = s["gt_bin"]; df["pred_bin"] = res[name]["probs"].argmax(1); df["routed_bin"] = r["routed_bin"]
        df["clip_outcome"] = r["outcome"]; df["disagreement"] = r["disagreement"]; df["p_correct"] = res[name]["p_correct"]
        df["is_confident"] = res[name]["p_correct"] >= cep.threshold
        df.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"), index=False)

    te_s, te_r = S["test"], res["test"]
    per_bin = per_bin_table(te_s["gt_bin"], te_r["probs"].argmax(1), te_r["route"]["clipped"], te_s["roll"], bins)
    per_bin.to_csv(os.path.join(out_dir, "per_bin_test.csv"), index=False)

    # ---- metrics row ---------------------------------------------------------
    m = {"tag": tag, "input_mode": input_mode, "n_bins": n_bins, "seed": seed, "target_mode": cfg["target_mode"],
         "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "pct_test_adjacent_to_train": adj["pct_test_adjacent_to_train"],
         "best_epoch": p1["best_epoch"], "epochs_run": p1["epochs_run"], "best_val_acc": p1["best_val_acc"],
         "temperature": cep.temperature, "threshold": cep.threshold, "threshold_found": cep.threshold_found,
         "val_mae3_at_threshold": cep.val_mae_conf, "val_coverage_at_threshold": cep.val_coverage_pct,
         "clip_rate_test_pct": 100.0 * np.mean(te_r["route"]["outcome"] == "clipped"),
         "fallback_rate_test_pct": 100.0 * np.mean(te_r["route"]["outcome"] == "fallback"),
         "n_empty_bins": int(sum(r is None for r in regs)),
         "params_total": p1["params"]["total"], "train_minutes": (time.time() - t0) / 60.0}
    for split in ("val", "test"):
        m.update({f"{split}_{k}": v for k, v in res[split]["metrics"].items()})
    m.update(temporal_metrics(manifest, te_s, te_r, cep, cfg))
    for mode, d in per_mode.items():
        for k in ("mae2_all_clipped", "mae3_confident", "coverage_pct", "aurc"):
            m[f"test_{k}__{mode}"] = d["res"]["test"]["metrics"][k]
    pd.DataFrame([m]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    # ---- figures ---------------------------------------------------------------
    plot_confidence_vs_error(te_r["p_correct"], circular_error_deg(te_r["route"]["clipped"], te_s["roll"]),
                             cep.threshold, n_bins, os.path.join(out_dir, "fig5_left_confidence_vs_error"))
    plot_per_bin_accuracy(per_bin, n_bins, os.path.join(out_dir, "fig5_right_per_bin_accuracy"))
    plot_risk_coverage(res, cep, cfg["target_mae_deg"], os.path.join(out_dir, "fig_risk_coverage"))
    plot_reliability(te_r["p_correct"], te_s["gt_bin"] == te_r["probs"].argmax(1),
                     te_r["probs"].max(1), te_s["gt_bin"] == te_r["probs"].argmax(1),
                     os.path.join(out_dir, "fig_reliability"))
    plot_scatter(te_s["roll"], te_r["route"]["clipped"], te_r["p_correct"] >= cep.threshold,
                 os.path.join(out_dir, "fig_actual_vs_predicted"))
    plot_prediction_trace(te_s["roll"], te_r["route"]["clipped"], te_r["p_correct"] >= cep.threshold,
                          cfg["sequence_plot_len"], os.path.join(out_dir, "fig6_prediction_trace"))
    plot_error_histogram(res, cfg["target_mae_deg"], os.path.join(out_dir, "fig_error_histograms"))
    plot_threshold_sweep(primary["sweep"], cep, cfg, os.path.join(out_dir, "fig_threshold_sweep_val"))

    # ---- model bundle ----------------------------------------------------------
    bundle_cfg = {k: cfg[k] for k in ("image_size", "embed_dim", "num_heads", "num_layers", "ff_dim", "dropout",
                                      "target_mode", "block_len", "train_frac", "val_frac", "data_fraction")}
    bundle_cfg.update({"n_bins": n_bins, "seed": seed, "input_mode": input_mode, "main_folder": cfg["main_folder"],
                       "soft_top_k": int(cfg.get("soft_top_k", 1)), "neighbor_overlap": int(cfg.get("neighbor_overlap", 0)),
                       "temporal_window": int(cfg.get("temporal_window", 0))})
    save_model_bundle(os.path.join(out_dir, "model_bundle"), model, regs, cep, normalizer, bundle_cfg)
    shutil.copy(os.path.join(out_dir, "split_manifest.csv"), os.path.join(out_dir, "model_bundle", "split_manifest.csv"))

    # ---- extra evaluations that reuse this run's frozen model ------------------
    extras = {}
    if cfg["run_controls"] and input_mode == "both":
        extras["shuffled_pnf"] = shuffled_pnf_test(cache, pnf_norm, model, te, device, regs, cep, bins, cfg, seed)
    if cfg["run_baselines"] and input_mode == "both":
        extras.update(baselines(S, bins, cfg, seed, device, cache, pnf_norm))
    if extras:
        pd.DataFrame([{"baseline": k, **v} for k, v in extras.items()]).to_csv(
            os.path.join(out_dir, "baselines_and_controls.csv"), index=False)
        m["extras"] = extras

    del cache
    torch.cuda.empty_cache()
    print(f"    done in {m['train_minutes']:.1f} min | test MAE_all {m['test_mae2_all_clipped']:.2f}  "
          f"MAE_conf {m['test_mae3_confident']:.2f} @ {m['test_coverage_pct']:.1f}%  AURC {m['test_aurc']:.2f}")
    return m


# ============================================================================
# Temporal post-processing evaluation (test blocks are consecutive frames)
# ============================================================================

def temporal_metrics(manifest, te_s, te_r, cep, cfg) -> Dict:
    """
    Evaluates causal in-block smoothing and the hold-last-confident fallback on
    the test split, and summarises the lengths of uncertain runs.
    """
    w = int(cfg.get("temporal_window", 0))
    sub = manifest.iloc[te_s["idx"]]
    conf = te_r["p_correct"] >= cep.threshold
    pred = te_r["route"]["clipped"]; true = te_s["roll"]
    out = {}
    runs = uncertain_run_lengths(conf, sub["block_id"].values, sub["frame_index"].values)
    out["uncertain_runs_n"] = int(len(runs))
    out["uncertain_run_median_frames"] = float(np.median(runs)) if len(runs) else 0.0
    out["uncertain_run_max_frames"] = int(runs.max()) if len(runs) else 0
    out["uncertain_run_mean_s"] = float(runs.mean() / 15.0) if len(runs) else 0.0
    if w <= 0:
        return out
    tp = temporal_postprocess(pred, conf, sub["block_id"].values, sub["frame_index"].values, w)
    out["test_mae_all_smoothed"] = float(circular_error_deg(tp["smoothed"], true).mean())
    out["test_mae_confident_smoothed"] = float(circular_error_deg(tp["smoothed"], true)[conf].mean()) if conf.any() else float("nan")
    out["test_mae_deploy_fallback"] = float(circular_error_deg(tp["fallback"], true).mean())
    out["test_pct_frames_filled_by_fallback"] = 100.0 * float(tp["fallback_filled"].mean())
    out["test_mae_deploy_fallback_smoothed"] = float(circular_error_deg(tp["smoothed_fallback"], true).mean())
    return out


# ============================================================================
# Controls and baselines
# ============================================================================

def shuffled_pnf_test(cache, pnf_norm, model, te, device, regs, cep, bins, cfg, seed) -> Dict:
    """Test-time control: PNFs permuted across test frames (images unchanged)."""
    rng = np.random.RandomState(seed + 1000)
    pn = pnf_norm.copy()
    pn[te] = pn[te][rng.permutation(len(te))]
    s = extract(cache, pnf_norm, model, te, device, 32, pnf_override=pn)
    probs = cep.calibrated_probs(s["logits"])
    route = route_and_predict(probs, s["features"], regs, bins, soft_top_k=int(cfg.get("soft_top_k", 1)))
    pc = cep.p_correct(probs, route["disagreement"])
    err = circular_error_deg(route["clipped"], s["roll"])
    conf = pc >= cep.threshold
    return {"mae_all": float(err.mean()), "mae_confident": float(err[conf].mean()) if conf.any() else float("nan"),
            "coverage_pct": 100.0 * conf.mean(),
            "phase1_accuracy": float((bins.index(s["roll"]) == probs.argmax(1)).mean()),
            "description": "Full model, test PNFs permuted across frames"}


def baselines(S, bins, cfg, seed, device, cache, pnf_norm) -> Dict[str, Dict]:
    """
    Matched baselines evaluated with the same circular all-frame metric:
      global_xgb_trained_features : single (cos,sin) XGBoost on the 304-dim features of this run
      conquer_only_imagenet       : single (cos,sin) XGBoost on frozen ImageNet ResNet-18 features + PNFs
      divide_only_mlp_head        : single MLP regression head on the frozen Phase 1 features
      bin_centre_predicted        : centre of the Phase 1 predicted bin
      bin_centre_oracle           : centre of the ground-truth bin (regression-free lower bound)
    """
    out = {}
    Xtr, ytr = S["train"]["features"], S["train"]["roll"]
    Xte, yte = S["test"]["features"], S["test"]["roll"]

    g = BinRegressor("cossin", cfg["xgb_params"], seed).fit(Xtr, ytr)
    out["global_xgb_trained_features"] = _mae_row(g.predict(Xte), yte, "Single XGBoost (cos,sin) on Phase 1 features, no routing")

    mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=1000, early_stopping=True,
                       random_state=seed)
    mlp.fit(Xtr, angle_to_cossin(ytr))
    p = mlp.predict(Xte)
    out["divide_only_mlp_head"] = _mae_row(cossin_to_angle(p[:, 0], p[:, 1]), yte,
                                           "Single MLP (cos,sin) head on frozen Phase 1 features")

    pred_bins = S["test"]["logits"].argmax(1)
    out["bin_centre_predicted"] = _mae_row(bins.centres()[pred_bins], yte, "Centre of predicted bin")
    out["bin_centre_oracle"] = _mae_row(bins.centres()[S["test"]["gt_bin"]], yte, "Centre of ground-truth bin")

    # Conquer-only: no Phase 1 training at all; ImageNet features + PNFs -> one regressor.
    if cache.images is not None:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained_backbone"] else None
            resnet = models.resnet18(weights=weights)
        except Exception as e:  # pretrained weights unavailable offline
            print(f"    conquer-only baseline skipped: {e}")
            return out
        resnet = resnet.to(device).eval()
        resnet.fc = torch.nn.Identity()
        def feats(idx):
            with torch.no_grad():
                f = [resnet(normalize_batch(cache.images[idx[s:s + 32]].to(device))).cpu().numpy()
                     for s in range(0, len(idx), 32)]
            return np.concatenate([np.concatenate(f), pnf_norm[idx]], axis=1)
        gi = BinRegressor("cossin", cfg["xgb_params"], seed).fit(feats(S["train"]["idx"]), ytr)
        out["conquer_only_imagenet"] = _mae_row(gi.predict(feats(S["test"]["idx"])), yte,
                                                "Single XGBoost (cos,sin) on frozen ImageNet ResNet-18 + PNFs")
    return out


def _mae_row(pred, true, desc) -> Dict:
    e = circular_error_deg(pred, true)
    return {"mae_all": float(e.mean()), "std_all": float(e.std()), "median_all": float(np.median(e)),
            "pct_within_10": 100.0 * float((e <= 10).mean()), "description": desc}


def per_bin_table(gt_bin, pred_bin, pred_roll, true_roll, bins: BinScheme) -> pd.DataFrame:
    rows = []
    err = circular_error_deg(pred_roll, true_roll)
    for i in range(bins.n_bins):
        m = gt_bin == i
        n = int(m.sum())
        acc = float((pred_bin[m] == i).mean()) if n else float("nan")
        # Wilson 95% interval for the per-bin accuracy
        if n:
            z = 1.96; ph = acc
            den = 1 + z ** 2 / n
            c = (ph + z ** 2 / (2 * n)) / den
            h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
            lo, hi = max(0.0, c - h), min(1.0, c + h)
        else:
            lo = hi = float("nan")
        rows.append({"bin": i, "range_deg": f"[{bins.lower(i):.1f}, {bins.upper(i):.1f})", "n_test": n,
                     "phase1_accuracy": acc, "ci95_low": lo, "ci95_high": hi,
                     "mae_in_bin": float(err[m].mean()) if n else float("nan")})
    return pd.DataFrame(rows)


# ============================================================================
# Figures (per run)
# ============================================================================

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=300)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def plot_learning_curves(curves, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(curves["epoch"], curves["train_loss"], label="Train"); ax[0].plot(curves["epoch"], curves["val_loss"], label="Validation")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(curves["epoch"], curves["train_acc"], label="Train"); ax[1].plot(curves["epoch"], curves["val_acc"], label="Validation")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Phase 1 accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    _save(fig, path)


def plot_confidence_vs_error(pc, err, thr, n_bins, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    conf = pc >= thr
    ax.scatter(pc[conf], err[conf], s=12, alpha=.45, label=f"Confident (n={conf.sum()})")
    ax.scatter(pc[~conf], err[~conf], s=12, alpha=.45, label=f"Uncertain (n={(~conf).sum()})")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"threshold = {thr:.2f}")
    ax.set_ylim(0, 180); ax.set_xlim(0, 1)
    ax.set_xlabel("Meta-classifier P(Phase 1 correct)"); ax.set_ylabel("Circular absolute error (deg)")
    ax.set_title(f"Confidence vs error, test split (N = {n_bins})"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, path)


def plot_per_bin_accuracy(per_bin, n_bins, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = per_bin["bin"].values; acc = per_bin["phase1_accuracy"].values
    yerr = np.vstack([acc - per_bin["ci95_low"].values, per_bin["ci95_high"].values - acc])
    yerr = np.clip(np.nan_to_num(yerr, nan=0.0), 0, None)
    ax.bar(x, acc, yerr=yerr, capsize=2, alpha=.85)
    for xi, n in zip(x, per_bin["n_test"].values):
        ax.text(xi, 0.02, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90, color="w")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Angular bin"); ax.set_ylabel("Phase 1 accuracy (test)")
    ax.set_title(f"Per-bin Phase 1 accuracy with 95% CI (N = {n_bins})"); ax.grid(axis="y", alpha=.3)
    _save(fig, path)


def plot_risk_coverage(res, cep, target, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, c in (("val", "tab:gray"), ("test", "tab:blue")):
        r = res[name]
        rc = risk_coverage_curve(_err_from_res(res, name), r["p_correct"])
        ax.plot(rc["coverage"] * 100, rc["risk_mae"], color=c, label=f"{name} (AURC = {area_under_risk_coverage(rc):.2f})")
    m = res["test"]["metrics"]
    ax.scatter([m["coverage_pct"]], [m["mae3_confident"]], color="tab:red", zorder=5,
               label=f"operating point ({m['coverage_pct']:.0f}%, {m['mae3_confident']:.1f} deg)")
    ax.axhline(target, color="k", ls=":", lw=1, label=f"target {target:.0f} deg")
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Selective risk: MAE of accepted frames (deg)")
    ax.set_title("Risk-coverage curve"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, path)


def _err_from_res(res, name):
    return res[name]["_err"]


def plot_reliability(pc, meta_correct, pmax, p1_correct, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, conf, corr, title in ((ax[0], pmax, p1_correct, "Phase 1 max probability (temperature-scaled)"),
                                 (ax[1], pc, meta_correct, "Meta-classifier P(correct)")):
        edges = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                xs.append(conf[m].mean()); ys.append(corr[m].mean()); ns.append(m.sum())
        a.plot([0, 1], [0, 1], "k--", lw=1)
        a.bar(xs, ys, width=0.08, alpha=.7)
        for x, y, n in zip(xs, ys, ns):
            a.text(x, y + .02, str(n), ha="center", fontsize=6)
        a.set_xlim(0, 1); a.set_ylim(0, 1.05); a.set_xlabel("Predicted confidence"); a.set_ylabel("Observed accuracy")
        a.set_title(title, fontsize=9); a.grid(alpha=.3)
    _save(fig, path)


def plot_scatter(true, pred, conf, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(true[conf], pred[conf], s=10, alpha=.5, label="Confident")
    ax.scatter(true[~conf], pred[~conf], s=10, alpha=.5, label="Uncertain")
    ax.plot([-180, 180], [-180, 180], "r--", lw=1)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("Ground-truth roll (deg)"); ax.set_ylabel("Predicted roll (deg)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Test split")
    _save(fig, path)


def plot_prediction_trace(true, pred, conf, n_show, path):
    """Test-set predictions over sample index (samples are not temporally consecutive)."""
    n = min(n_show, len(true)); i = np.arange(n)
    err = circular_error_deg(pred[:n], true[:n])
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(i, true[:n], label="Ground truth", lw=1.2)
    ax[0].plot(i, pred[:n], label="Predicted", lw=1.0, alpha=.8)
    ax[0].scatter(i[~conf[:n]], pred[:n][~conf[:n]], s=14, color="tab:red", zorder=5, label="Flagged uncertain")
    ax[0].set_ylabel("Roll (deg)"); ax[0].set_ylim(-190, 190); ax[0].legend(fontsize=8, ncol=3); ax[0].grid(alpha=.3)
    ax[1].bar(i, err, width=1.0, color="k", alpha=.7)
    ax[1].set_ylabel("Circular error (deg)"); ax[1].set_xlabel("Test sample index (shuffled, not consecutive)")
    ax[1].set_ylim(0, 180); ax[1].grid(alpha=.3)
    _save(fig, path)


def plot_error_histogram(res, target, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    e_all = _err_from_res(res, "test"); conf = res["test"]["p_correct"] >= res["test"]["_thr"]
    ax.hist(e_all, bins=36, range=(0, 180), alpha=.5, label=f"All frames (MAE {e_all.mean():.1f} deg)")
    if conf.any():
        ax.hist(e_all[conf], bins=36, range=(0, 180), alpha=.7, label=f"Confident (MAE {e_all[conf].mean():.1f} deg)")
    ax.axvline(target, color="k", ls="--", lw=1, label=f"{target:.0f} deg target")
    ax.set_xlabel("Circular absolute error (deg)"); ax.set_ylabel("Count"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, path)


def plot_threshold_sweep(sweep, cep, cfg, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax2 = ax.twinx()
    ax.plot(sweep["threshold"], sweep["val_mae_conf"], "b-o", ms=2.5, label="Validation MAE (confident)")
    ax2.plot(sweep["threshold"], sweep["val_coverage_pct"], "g--s", ms=2.5, label="Validation coverage")
    ax.axhline(cfg["target_mae_deg"], color="r", ls="--", lw=1); ax2.axhline(cfg["min_coverage_pct"], color="g", ls=":", lw=1)
    ax.axvline(cep.threshold, color="k", ls="--", lw=1, label=f"selected = {cep.threshold:.2f}")
    ax.set_xlabel("Meta-classifier threshold"); ax.set_ylabel("MAE (deg)", color="b"); ax2.set_ylabel("Coverage (%)", color="g")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.grid(alpha=.3)
    _save(fig, path)


# ============================================================================
# Summary figures and tables
# ============================================================================

def plot_n_sweep(df: pd.DataFrame, path: str, target: float):
    """Figure 4: validation-based N sweep. Missing MAE3 values are shown as missing."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = df["n_bins"].values
    for a, col, lab in ((ax[0], "val_mae3_confident", "Validation MAE, confident (deg)"),
                        (ax[1], "val_coverage_pct", "Validation coverage (%)"),
                        (ax[2], "val_mae2_all_clipped", "Validation MAE, all frames (deg)")):
        y = df[col].values.astype(float)
        ok = ~np.isnan(y) & df["threshold_found"].values.astype(bool)
        a.plot(x[ok], y[ok], "o-", label="threshold found")
        bad = ~ok & ~np.isnan(y)
        a.plot(x[bad], y[bad], "x", color="tab:red", label="criteria not met")
        miss = np.isnan(y)
        for xi in x[miss]:
            a.text(xi, a.get_ylim()[0], "n/a", ha="center", va="bottom", fontsize=7, color="tab:red")
        a.set_xlabel("Number of bins N"); a.set_ylabel(lab); a.grid(alpha=.3); a.set_xticks(x)
    ax[0].axhline(target, color="k", ls=":", lw=1); ax[0].legend(fontsize=7)
    sel = df.loc[df["selected"], "n_bins"].values
    if len(sel):
        for a in ax: a.axvline(sel[0], color="tab:green", alpha=.3, lw=6)
    fig.suptitle("N sweep on the validation split (selection criterion marked in green)")
    _save(fig, path)


def plot_seed_stability(df: pd.DataFrame, path: str, target: float):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df)); w = .38
    ax[0].bar(x - w / 2, df["test_mae2_all_clipped"], w, label="All frames")
    ax[0].bar(x + w / 2, df["test_mae3_confident"], w, label="Confident")
    ax[0].axhline(target, color="k", ls=":", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{s}{'*' if sel else ''}" for s, sel in zip(df["seed"], df["selected_checkpoint"])])
    ax[0].set_xlabel("Seed (* = released checkpoint, chosen on validation)"); ax[0].set_ylabel("Test MAE (deg)"); ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)
    ax[1].bar(x, df["test_coverage_pct"]); ax[1].set_xticks(x); ax[1].set_xticklabels(df["seed"])
    ax[1].set_xlabel("Seed"); ax[1].set_ylabel("Test coverage (%)"); ax[1].grid(axis="y", alpha=.3)
    _save(fig, path)


def plot_ablation(df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    d = df.sort_values("mae_all", ascending=False)
    ax.barh(d["method"], d["mae_all"], xerr=d.get("std_all"), capsize=2)
    ax.set_xlabel("Test MAE, all frames, circular (deg)"); ax.grid(axis="x", alpha=.3)
    _save(fig, path)


# ============================================================================
# Orchestration
# ============================================================================

def selection_score(df: pd.DataFrame, cfg: Dict) -> pd.Series:
    """
    Validation-only selection score (lower is better).
      'composite': MAE_all * MAE_conf * AURC * (100 - coverage%), a joint objective
                   that penalises error and abstention together.
      'mae_conf' : confident-subset MAE only.
    """
    if cfg.get("n_selection", "composite") == "composite":
        return (df["val_mae2_all_clipped"] * df["val_mae3_confident"] * df["val_aurc"]
                * (100.0 - df["val_coverage_pct"]))
    return df["val_mae3_confident"]


def select_best(df: pd.DataFrame, key: str, cfg: Dict) -> int:
    """
    Apply the validation selection rule to a table of runs (one row per N or
    per seed) and return the value of `key` ('n_bins' or 'seed') for the best
    row. Only rows whose threshold search succeeded with coverage >= the
    minimum are eligible; if none qualify, fall back to the lowest validation
    all-frame MAE. Test columns are never consulted.
    """
    ok = df[(df["threshold_found"].astype(bool)) & (df["val_coverage_pct"] >= cfg["min_coverage_pct"])].copy()
    if len(ok):
        ok["_score"] = selection_score(ok, cfg)
        return int(ok.loc[ok["_score"].idxmin(), key])
    return int(df.loc[df["val_mae2_all_clipped"].idxmin(), key])


def select_n(sweep: pd.DataFrame, cfg: Dict) -> int:
    return select_best(sweep, "n_bins", cfg)


def main(cfg: Dict) -> None:
    if cfg["smoke_test"]:
        cfg = {**cfg, **cfg["smoke_overrides"]}
    root = cfg["results_root"]
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config_used.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if k != "threshold_grid"}, f, indent=2, default=str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rows_all: List[Dict] = []

    # ---- Stage A: N sweep (validation-based selection) ------------------------
    sweep_rows = []
    for n in cfg["n_values"]:
        m = run_experiment(n, cfg["sweep_seed"], cfg, device, os.path.join(root, "A_n_sweep", f"N{n:02d}_seed{cfg['sweep_seed']}"), tag="n_sweep")
        sweep_rows.append(m); rows_all.append(m)
    sweep = pd.DataFrame([{k: v for k, v in r.items() if k != "extras"} for r in sweep_rows])
    best_n = select_n(sweep, cfg)
    sweep["selected"] = sweep["n_bins"] == best_n
    sweep["val_selection_score"] = selection_score(sweep, cfg)
    cols = ["n_bins", "selected", "val_selection_score", "threshold_found", "threshold", "temperature", "n_empty_bins",
            "val_mae2_all_clipped", "val_mae3_confident", "val_coverage_pct", "val_n_confident", "val_aurc",
            "test_mae2_all_clipped", "test_mae3_confident", "test_coverage_pct", "test_n_confident", "test_aurc",
            "test_r2_circular", "test_phase1_accuracy"]
    sweep[cols].round(3).to_csv(os.path.join(root, "table1_n_sweep.csv"), index=False)
    plot_n_sweep(sweep, os.path.join(root, "fig4_n_sweep"), cfg["target_mae_deg"])
    print(f"\n>>> Selected N = {best_n} (validation criterion)")

    # ---- Stage B: seed stability at the selected N ---------------------------
    stab_rows = []
    for seed in cfg["stability_seeds"]:
        if seed == cfg["sweep_seed"]:
            m = next(r for r in sweep_rows if r["n_bins"] == best_n)
        else:
            m = run_experiment(best_n, seed, cfg, device, os.path.join(root, "B_seed_stability", f"N{best_n:02d}_seed{seed}"), tag="stability")
            rows_all.append(m)
        stab_rows.append(m)
    stab = pd.DataFrame([{k: v for k, v in r.items() if k != "extras"} for r in stab_rows])
    # Released checkpoint: best validation confident MAE among seeds meeting the coverage criterion,
    # otherwise best validation all-frame MAE.
    chosen_seed = select_best(stab, "seed", cfg)
    stab["selected_checkpoint"] = stab["seed"] == chosen_seed
    stab["val_selection_score"] = selection_score(stab, cfg)
    scols = ["seed", "selected_checkpoint", "val_selection_score", "threshold_found", "threshold", "best_val_acc",
             "val_mae2_all_clipped", "val_mae3_confident", "val_coverage_pct",
             "test_mae1_all_raw", "test_mae2_all_clipped", "test_mae3_confident", "test_std3_confident",
             "test_coverage_pct", "test_n_confident", "test_n_uncertain", "test_aurc", "test_r2_circular",
             "test_phase1_accuracy", "test_ece_meta", "test_brier_meta", "test_filter_auroc",
             "test_filter_large_error_accept_rate", "clip_rate_test_pct",
             "test_mae_all_smoothed", "test_mae_deploy_fallback", "test_mae_deploy_fallback_smoothed",
             "test_pct_frames_filled_by_fallback", "uncertain_run_median_frames", "uncertain_run_max_frames"]
    stab[[c for c in scols if c in stab.columns]].round(3).to_csv(os.path.join(root, "table_seed_stability.csv"), index=False)
    agg = {}
    for c in ("test_mae2_all_clipped", "test_mae3_confident", "test_coverage_pct", "test_aurc", "test_r2_circular", "test_phase1_accuracy"):
        v = stab[c].astype(float)
        agg[c] = {"mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0, "min": v.min(), "max": v.max(),
                  "released_checkpoint": float(stab.loc[stab["selected_checkpoint"], c].iloc[0])}
    pd.DataFrame(agg).T.round(3).to_csv(os.path.join(root, "table_seed_summary.csv"))
    plot_seed_stability(stab, os.path.join(root, "fig_seed_stability"), cfg["target_mae_deg"])
    print(f">>> Released checkpoint: seed {chosen_seed} (validation criterion)")

    # Copy the released bundle to a dedicated folder.
    src_dir = os.path.join(root, "A_n_sweep", f"N{best_n:02d}_seed{chosen_seed}") if chosen_seed == cfg["sweep_seed"] \
        else os.path.join(root, "B_seed_stability", f"N{best_n:02d}_seed{chosen_seed}")
    rel = os.path.join(root, "released_model")
    if os.path.exists(rel):
        shutil.rmtree(rel)
    shutil.copytree(os.path.join(src_dir, "model_bundle"), rel)
    for f in ("predictions_test.csv", "metrics.csv", "per_bin_test.csv"):
        shutil.copy(os.path.join(src_dir, f), os.path.join(rel, f))

    # ---- Stage C: input-modality controls at the selected N -------------------
    if cfg["run_controls"]:
        ctrl_rows = []
        for mode in ("image", "pnf"):
            m = run_experiment(best_n, chosen_seed, cfg, device, os.path.join(root, "C_controls", f"{mode}_only_N{best_n:02d}_seed{chosen_seed}"),
                               input_mode=mode, tag="control")
            rows_all.append(m)
            ctrl_rows.append({"method": f"{mode}-only DAC-RViT", "mae_all": m["test_mae2_all_clipped"],
                              "std_all": m["test_std2_all_clipped"], "mae_confident": m["test_mae3_confident"],
                              "coverage_pct": m["test_coverage_pct"], "phase1_accuracy": m["test_phase1_accuracy"]})
        full = next(r for r in stab_rows if r["seed"] == chosen_seed)
        ctrl_rows.append({"method": "Full DAC-RViT (image + PNF)", "mae_all": full["test_mae2_all_clipped"],
                          "std_all": full["test_std2_all_clipped"], "mae_confident": full["test_mae3_confident"],
                          "coverage_pct": full["test_coverage_pct"], "phase1_accuracy": full["test_phase1_accuracy"]})
        sh = full.get("extras", {}).get("shuffled_pnf")
        if sh:
            ctrl_rows.append({"method": "Full DAC-RViT, shuffled PNFs at test", "mae_all": sh["mae_all"], "std_all": np.nan,
                              "mae_confident": sh["mae_confident"], "coverage_pct": sh["coverage_pct"],
                              "phase1_accuracy": sh["phase1_accuracy"]})
        pd.DataFrame(ctrl_rows).round(3).to_csv(os.path.join(root, "table_controls.csv"), index=False)

    # ---- Stage D: ablation table from the released checkpoint's baselines ------
    if cfg["run_baselines"]:
        full = next(r for r in stab_rows if r["seed"] == chosen_seed)
        ab = []
        names = {"conquer_only_imagenet": "Conquer-only (N = 1, ImageNet features + PNF)",
                 "global_xgb_trained_features": "Global XGBoost on Phase 1 features (no routing)",
                 "divide_only_mlp_head": "Divide-only (single MLP head on Phase 1 features)",
                 "bin_centre_predicted": "Predicted-bin centre",
                 "bin_centre_oracle": "Oracle-bin centre"}
        for k, v in full.get("extras", {}).items():
            if k in names:
                ab.append({"method": names[k], "mae_all": v["mae_all"], "std_all": v.get("std_all", np.nan),
                           "median_all": v.get("median_all", np.nan), "pct_within_10": v.get("pct_within_10", np.nan)})
        ab.append({"method": f"Full DAC-RViT (N = {best_n})", "mae_all": full["test_mae2_all_clipped"],
                   "std_all": full["test_std2_all_clipped"], "median_all": full["test_median_err_all"],
                   "pct_within_10": full["test_pct_within_target_all"]})
        # Seed-averaged row for the full model
        ab.append({"method": f"Full DAC-RViT (N = {best_n}), mean over {len(stab)} seeds",
                   "mae_all": agg["test_mae2_all_clipped"]["mean"], "std_all": agg["test_mae2_all_clipped"]["std"],
                   "median_all": np.nan, "pct_within_10": np.nan})
        abdf = pd.DataFrame(ab)
        abdf.round(3).to_csv(os.path.join(root, "table2_ablation.csv"), index=False)
        plot_ablation(abdf, os.path.join(root, "fig_ablation"))

    # ---- all runs -----------------------------------------------------------------
    pd.DataFrame([{k: v for k, v in r.items() if k != "extras"} for r in rows_all]).to_csv(
        os.path.join(root, "all_runs.csv"), index=False)
    print(f"\nAll results written to: {os.path.abspath(root)}")


if __name__ == "__main__":
    main(CONFIG)
