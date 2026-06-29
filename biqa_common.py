"""
Shared utilities for the BIQA frozen-fusion + LoRA-adaptation experiments.

Covers: dataset loading (via IQA-PyTorch meta-info CSVs), NSS feature
extraction, frozen SigLIP/CLIP-H five-crop embedding, image-level and
reference-level splitting, the MLP regression head, and the hybrid loss.

The datasets are loaded through the meta-info files distributed with the
IQA-PyTorch toolbox, which provide, per image, a reference identity and a
quality score. This makes the reference-level split a simple group-by on the
reference column.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------------------
# Dataset specification
# --------------------------------------------------------------------------
# Each entry maps a dataset key to:
#   meta   : meta-info CSV filename (from chaofengc/IQA-Toolbox-Datasets-metainfo)
#   ref    : column holding the reference identity
#   dist   : column holding the distorted-image path (relative to img_root)
#   score  : column holding the quality score
#   invert : if True, score is a DMOS (higher = worse) and is inverted
#   authentic : if True, no reference structure (image-level == reference-level)

DATASETS = {
    "koniq":   dict(meta="meta_info_KonIQ10kDataset.csv",   ref=None,        dist="img_name",  score="mos",              invert=False, authentic=True),
    "liveitw": dict(meta="meta_info_LIVEChallengeDataset.csv", ref=None,     dist="img_name",  score="mos",              invert=False, authentic=True),
    "kadid":   dict(meta="meta_info_KADID10kDataset.csv",   ref="ref_name",  dist="dist_name", score="dmos",             invert=True,  authentic=False),
    "tid2013": dict(meta="meta_info_TID2013Dataset.csv",    ref="ref_name",  dist="dist_name", score="mos",              invert=False, authentic=False),
    "csiq":    dict(meta="meta_info_CSIQDataset.csv",       ref="ref_name",  dist="dist_name", score="dmos",             invert=True,  authentic=False),
    "livemd":  dict(meta="meta_info_LIVEMDDataset.csv",     ref="ref_name",  dist="dist_name", score="dmos",             invert=True,  authentic=False),
    "pipal":   dict(meta="meta_info_PIPALDataset.csv",      ref="hq_name",   dist="dist_name", score="scaled_elo_score", invert=False, authentic=False),
}

SIGLIP_CKPT = "google/siglip-so400m-patch14-384"


def load_dataset(key, meta_dir, img_root):
    """Return (paths, refs, y) for a dataset.

    paths : list of absolute image paths
    refs  : np.ndarray of reference identities (per image); for authentic
            datasets each image is its own reference
    y     : np.ndarray of quality scores (higher = better)
    """
    spec = DATASETS[key]
    df = pd.read_csv(os.path.join(meta_dir, spec["meta"]))
    df = df.dropna(subset=[spec["dist"], spec["score"]])

    paths, refs, scores = [], [], []
    for _, r in df.iterrows():
        p = os.path.join(img_root, str(r[spec["dist"]]))
        if not os.path.exists(p):
            continue
        paths.append(p)
        if spec["authentic"] or spec["ref"] is None:
            refs.append(str(r[spec["dist"]]))      # each image its own group
        else:
            refs.append(str(r[spec["ref"]]))
        scores.append(float(r[spec["score"]]))

    y = np.asarray(scores, dtype="float32")
    if spec["invert"]:
        y = (y.max() - y)                          # DMOS -> quality (higher = better)
    return paths, np.asarray(refs), y


# --------------------------------------------------------------------------
# Splitting protocols
# --------------------------------------------------------------------------
def image_level_split(n, seed=42, test_frac=0.2):
    """Random split over individual distorted images (the conventional protocol)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int((1 - test_frac) * n)
    return idx[:cut], idx[cut:]


def reference_level_split(refs, seed=42, test_frac=0.2):
    """Group split: all distorted versions of a reference go to one partition.

    Guarantees zero reference overlap between train and test.
    """
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(refs)))
    rng.shuffle(uniq)
    n_test = max(1, int(test_frac * len(uniq)))
    test_refs = set(uniq[:n_test])
    test = np.array([i for i, r in enumerate(refs) if r in test_refs])
    train = np.array([i for i, r in enumerate(refs) if r not in test_refs])
    assert len(set(refs[train]) & set(refs[test])) == 0, "reference overlap!"
    return train, test


# --------------------------------------------------------------------------
# Frozen VLM five-crop embedding
# --------------------------------------------------------------------------
def five_crop_boxes(W, H, frac=0.85):
    s = int(frac * min(W, H))
    return [
        ((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2),  # center
        (0, 0, s, s), (W - s, 0, W, s), (0, H - s, s, H), (W - s, H - s, W, H),  # corners
    ]


@torch.no_grad()
def embed_image(path, model, processor, device):
    """L2-normalized average of five-crop SigLIP/CLIP embeddings for one image."""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    embs = []
    for box in five_crop_boxes(W, H):
        crop = img.crop(box)
        px = processor(images=crop, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device, dtype=torch.float16):
            e = model(pixel_values=px).pooler_output[0].float()
        embs.append(e / (e.norm() + 1e-8))
    return torch.stack(embs).mean(0).cpu().numpy()


# --------------------------------------------------------------------------
# MLP regression head and hybrid loss
# --------------------------------------------------------------------------
def make_head(in_dim, p_drop=0.3):
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(p_drop),
        nn.Linear(512, 256), nn.GELU(), nn.Dropout(p_drop),
        nn.Linear(256, 1),
    )


def hybrid_loss(pred, target, w_plcc=0.5):
    """MSE + (1 - PLCC). A lightweight rank-aware regression loss."""
    a = pred - pred.mean()
    b = target - target.mean()
    plcc = (a * b).sum() / (a.norm() * b.norm() + 1e-8)
    return F.mse_loss(pred, target) + w_plcc * (1 - plcc)
