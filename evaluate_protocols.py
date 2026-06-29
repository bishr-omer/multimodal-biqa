"""
Evaluate a frozen-feature dataset under both splitting protocols and report the
inflation gap (image-level minus reference-level SROCC).

This reproduces the leakage analysis: image-level splitting inflates scores
through content overlap; reference-level splitting removes it.

Usage:
    python evaluate_protocols.py --features features/kadid.npz
"""
import argparse, numpy as np
from biqa_common import image_level_split, reference_level_split
from train_head import train_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y, refs = d["X"], d["y"], d["refs"]
    n_refs = len(set(refs))
    print(f"{len(X)} images, {n_refs} references")

    tr_i, te_i = image_level_split(len(X), seed=args.seed)
    img_s, _ = train_eval(X, y, tr_i, te_i, epochs=args.epochs)

    tr_r, te_r = reference_level_split(refs, seed=args.seed)
    ref_s, _ = train_eval(X, y, tr_r, te_r, epochs=args.epochs)

    print(f"\nimage-level     SROCC: {img_s:.4f}")
    print(f"reference-level SROCC: {ref_s:.4f}")
    print(f"inflation gap:         {img_s - ref_s:.4f}  ({n_refs} refs)")


if __name__ == "__main__":
    main()
