"""
Reproduce the reported test-split results from a released model bundle.

The bundle contains the split manifest used for training. This script runs the
released model on exactly those test frames and recomputes every reported
metric with the circular error definition. Numbers must match
released_model/metrics.csv.

Usage
    python evaluate_test.py --bundle ../released_model --data path/to/Extracted_Frames_Roll_X_Y_4_ViT
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dac_rvit_core import (  # noqa: E402
    BinScheme, DACRViTPredictor, circular_error_deg, seed_everything, summarize_predictions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--data", required=True, help="root folder with recording subfolders")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="evaluation_output")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    seed_everything(0)
    os.makedirs(a.out, exist_ok=True)
    pred = DACRViTPredictor(a.bundle, device=a.device)
    bins = BinScheme(pred.bins.n_bins)
    man = pd.read_csv(os.path.join(a.bundle, "split_manifest.csv"))
    man = man[man["split"] == a.split].reset_index(drop=True)
    # Re-root the stored paths onto the local data folder.
    man["path"] = [os.path.join(a.data, rec, f"frame_{int(i):04d}.png")
                   for rec, i in zip(man["recording"], man["frame_index"])]
    missing = [p for p in man["path"] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames not found, e.g. {missing[0]}")

    pnf = man[["pnf_0", "pnf_1", "pnf_2", "pnf_3"]].values.astype(np.float32)
    r = pred.predict(man["path"].tolist(), pnf)
    true = man["roll"].values
    gt_bin = bins.index(true)
    pred_bin = r["probs"].argmax(1)
    metrics = summarize_predictions(true, r["roll_raw_deg"], r["roll_deg"], r["p_correct"],
                                    pred.cep.threshold, pred.cep.target_mae_deg, gt_bin, pred_bin,
                                    r["probs"], bins.n_bins)
    out = man[["recording", "frame_index", "split"]].copy()
    out["true_roll"] = true; out["pred_roll"] = r["roll_deg"]; out["pred_roll_raw"] = r["roll_raw_deg"]
    out["error_circular"] = circular_error_deg(r["roll_deg"], true)
    out["gt_bin"] = gt_bin; out["pred_bin"] = pred_bin; out["routed_bin"] = r["bin"]
    out["p_correct"] = r["p_correct"]; out["is_confident"] = r["is_confident"]
    out.to_csv(os.path.join(a.out, f"predictions_{a.split}.csv"), index=False)
    with open(os.path.join(a.out, f"metrics_{a.split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{a.split} split, n = {metrics['n']}")
    print(f"  MAE all frames (clipped)   : {metrics['mae2_all_clipped']:.2f} deg")
    print(f"  MAE confident subset       : {metrics['mae3_confident']:.2f} deg at {metrics['coverage_pct']:.1f}% coverage")
    print(f"  AURC                        : {metrics['aurc']:.2f}")
    print(f"  Phase 1 accuracy            : {metrics['phase1_accuracy']:.3f}")
    print(f"  Circular R2                 : {metrics['r2_circular']:.3f}")
    print(f"Outputs written to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
