"""
Single-frame / batch inference with a released DAC-RViT model bundle.

Usage
    python predict.py --bundle ../released_model --image frame_0012.png \
        --lo 60.40 46.65 --td 70.74 50.06

    python predict.py --bundle ../released_model --csv frames.csv
        (csv columns: path, X_Local_Origin, Y_Local_Origin, X_Tip_Displacement, Y_Tip_Displacement)

Input images are expected to be the preprocessed 224x224 crops produced by the
upstream inpainting + attention-guided cropping step. Output per frame:
roll angle (deg), raw regressor angle, routed bin, P(Phase 1 correct) and the
confident/uncertain flag at the released threshold.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dac_rvit_core import DACRViTPredictor, PNF_COLUMNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="path to released_model/")
    ap.add_argument("--image", help="single preprocessed frame (png)")
    ap.add_argument("--lo", nargs=2, type=float, metavar=("X", "Y"), help="local-origin coordinates")
    ap.add_argument("--td", nargs=2, type=float, metavar=("X", "Y"), help="tip coordinates")
    ap.add_argument("--csv", help="csv with columns: path + PNF columns")
    ap.add_argument("--out", default=None, help="optional output csv")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    a = ap.parse_args()

    pred = DACRViTPredictor(a.bundle, device=a.device)

    if a.csv:
        df = pd.read_csv(a.csv)
        paths = df["path"].tolist()
        pnf = df[PNF_COLUMNS].values.astype(np.float32)
    elif a.image and a.lo and a.td:
        paths = [a.image]
        pnf = np.array([[a.lo[0], a.lo[1], a.td[0], a.td[1]]], dtype=np.float32)
    else:
        ap.error("provide --csv, or --image with --lo and --td")

    r = pred.predict(paths, pnf)
    out = pd.DataFrame({"path": paths, "roll_deg": r["roll_deg"], "roll_raw_deg": r["roll_raw_deg"],
                        "bin": r["bin"], "p_correct": r["p_correct"], "is_confident": r["is_confident"]})
    if a.out:
        out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
