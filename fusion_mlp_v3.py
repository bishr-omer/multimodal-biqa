"""
fusion_mlp_v3.py - KonIQ-10k MLP fusion with ranking loss
Streams: NSS (138) + CLIP-L (768) + CLIP-H (1024) + SigLIP (1152) + DINOv2 (1024) + LLaVA (4096)
Regressor: 3-layer MLP, hybrid loss (MSE + PLCC + ranking)
5-fold CV, dumps per-image predictions to CSV
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")

# Output dir for CSV dumps
os.makedirs("results/predictions", exist_ok=True)

# Load features
def load(path):
    return np.load(path, allow_pickle=True).item()

print("Loading features...")
nss_data    = load("results/koniq_mvg_features.npy")
dino_data   = load("results/koniq_dino_features.npy")
clip_l_data = load("results/koniq_clip_large_features.npy")
clip_h_data = load("results/koniq_clip_h14_features.npy")
siglip_data = load("results/koniq_siglip_features.npy")
llava_data  = load("results/koniq_llava_features.npy")

# Load MOS
df = pd.read_csv("data/koniq10k/koniq10k_scores.csv").set_index("image_name")

# Build dicts
nss_dict    = dict(zip(nss_data["names"],    nss_data["features"]))
dino_dict   = dict(zip(dino_data["names"],   dino_data["features"]))
clip_l_dict = dict(zip(clip_l_data["names"], clip_l_data["features"]))
clip_h_dict = dict(zip(clip_h_data["names"], clip_h_data["features"]))
siglip_dict = dict(zip(siglip_data["names"], siglip_data["features"]))
llava_feat_dict  = dict(zip(llava_data["names"], llava_data["features"]))
llava_score_dict = dict(zip(llava_data["names"], llava_data["scores"]))

# Align
common = (set(nss_dict) & set(dino_dict) & set(clip_l_dict)
          & set(clip_h_dict) & set(siglip_dict) & set(llava_feat_dict)
          & set(df.index))
rows = sorted(common)
print(f"Aligned samples: {len(rows)}")

X_nss    = np.array([nss_dict[n]    for n in rows]).astype(np.float32)
X_dino   = np.array([dino_dict[n]   for n in rows]).astype(np.float32)
X_clip_l = np.array([clip_l_dict[n] for n in rows]).astype(np.float32)
X_clip_h = np.array([clip_h_dict[n] for n in rows]).astype(np.float32)
X_siglip = np.array([siglip_dict[n] for n in rows]).astype(np.float32)
X_llava_feat  = np.array([llava_feat_dict[n]  for n in rows]).astype(np.float32)
X_llava_score = np.array([[llava_score_dict[n]] for n in rows]).astype(np.float32)
y = np.array([df.loc[n, "MOS"] for n in rows]).astype(np.float32)
names = np.array(rows)

print(f"NSS:    {X_nss.shape}")
print(f"DINOv2: {X_dino.shape}")
print(f"CLIP-L: {X_clip_l.shape}")
print(f"CLIP-H: {X_clip_h.shape}")
print(f"SigLIP: {X_siglip.shape}")
print(f"LLaVA:  {X_llava_feat.shape}")


class MLPHead(nn.Module):
    """3-layer MLP for quality regression."""
    def __init__(self, in_dim, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def plcc_loss(y_pred, y_true, eps=1e-8):
    """Negative PLCC as loss. Optimizing PLCC directly helps both PLCC and SROCC."""
    yp = y_pred - y_pred.mean()
    yt = y_true - y_true.mean()
    num = (yp * yt).sum()
    den = torch.sqrt((yp ** 2).sum() * (yt ** 2).sum() + eps)
    return 1.0 - num / den


def ranking_loss(y_pred, y_true, margin=0.0):
    """
    Pairwise ranking loss (fidelity-style).
    For each pair (i, j) with y_true[i] > y_true[j], encourage y_pred[i] > y_pred[j].
    """
    n = y_pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=y_pred.device)
    # All pairs
    pred_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
    true_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)
    # Mask: only pairs where true labels actually differ
    mask = (true_diff.abs() > 1e-3).float()
    # Sign of true difference
    sign = torch.sign(true_diff)
    # We want pred_diff to have the same sign as true_diff
    # Use logistic loss: -log(sigmoid(sign * pred_diff))
    loss = torch.nn.functional.softplus(-sign * pred_diff + margin) * mask
    return loss.sum() / (mask.sum() + 1e-8)


def hybrid_loss(y_pred, y_true, w_mse=1.0, w_plcc=0.5, w_rank=0.5):
    mse = nn.functional.mse_loss(y_pred, y_true)
    pl  = plcc_loss(y_pred, y_true)
    rk  = ranking_loss(y_pred, y_true)
    return w_mse * mse + w_plcc * pl + w_rank * rk, (mse.item(), pl.item(), rk.item())


def preprocess_streams(streams_dict, train_idx, test_idx, pca_threshold=1000, pca_components=256):
    """StandardScaler per stream + PCA for high-dim streams. Fit on train only."""
    train_parts, test_parts = [], []
    for name, X in streams_dict.items():
        X_tr, X_te = X[train_idx], X[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        if X_tr.shape[1] > pca_threshold:
            n_comp = min(pca_components, X_tr.shape[1], X_tr.shape[0] - 1)
            pca = PCA(n_components=n_comp, whiten=True, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        train_parts.append(X_tr.astype(np.float32))
        test_parts.append(X_te.astype(np.float32))
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def train_mlp(X_train, y_train, X_test, y_test,
              epochs=80, batch_size=128, lr=5e-4, weight_decay=1e-4,
              hidden1=512, hidden2=256, dropout=0.3, verbose=False):
    """Train MLP with hybrid loss and cosine LR schedule. Returns predictions on test."""
    in_dim = X_train.shape[1]
    model = MLPHead(in_dim, hidden1, hidden2, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train).to(device)
    X_te_t = torch.from_numpy(X_test).to(device)

    n = X_tr_t.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss, _ = hybrid_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        if verbose and (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1}/{epochs}  loss={np.mean(ep_losses):.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    return y_pred


def evaluate_cv(streams_dict, y, names, label, n_splits=5, save_csv=True):
    """5-fold CV with MLP head."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    sroccs, plccs, kroccs = [], [], []
    all_preds = np.zeros_like(y)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y), 1):
        X_train, X_test = preprocess_streams(streams_dict, train_idx, test_idx)
        y_pred = train_mlp(X_train, y[train_idx], X_test, y[test_idx])
        all_preds[test_idx] = y_pred

        y_true = y[test_idx]
        sr = stats.spearmanr(y_true,  y_pred).statistic
        pl = stats.pearsonr(y_true,   y_pred)[0]
        kr = stats.kendalltau(y_true, y_pred).statistic
        sroccs.append(sr); plccs.append(pl); kroccs.append(kr)
        print(f"    fold {fold_i}/{n_splits}  SROCC={sr:.4f}  PLCC={pl:.4f}", flush=True)

    if save_csv:
        safe_label = label.replace(" ", "_").replace("+", "and").replace("/", "-")
        csv_path = f"results/predictions/koniq_{safe_label}.csv"
        pd.DataFrame({
            "image_name": names,
            "true_mos": y,
            "pred_mos": all_preds,
        }).to_csv(csv_path, index=False)
        print(f"    -> saved per-image preds: {csv_path}")

    return (np.mean(sroccs), np.std(sroccs),
            np.mean(plccs),  np.std(plccs),
            np.mean(kroccs), np.std(kroccs))


# Fusion variants
variants = {
    # Single streams (sanity)
    "NSS only":              {"nss": X_nss},
    "CLIP-L only":           {"clip_l": X_clip_l},
    "CLIP-H only":           {"clip_h": X_clip_h},
    "SigLIP only":           {"siglip": X_siglip},
    "DINOv2 only":           {"dino": X_dino},
    "LLaVA-feat only":       {"llava_feat": X_llava_feat},

    # Pairs with NSS (ablation)
    "NSS + CLIP-H":          {"nss": X_nss, "clip_h": X_clip_h},
    "NSS + SigLIP":          {"nss": X_nss, "siglip": X_siglip},
    "NSS + DINOv2":          {"nss": X_nss, "dino": X_dino},
    "NSS + LLaVA-feat":      {"nss": X_nss, "llava_feat": X_llava_feat},

    # Multi-VLM no NSS (ablation)
    "SigLIP + CLIP-H + DINOv2": {
        "siglip": X_siglip, "clip_h": X_clip_h, "dino": X_dino,
    },

    # Three-stream fusion candidates
    "NSS + SigLIP + CLIP-H": {
        "nss": X_nss, "siglip": X_siglip, "clip_h": X_clip_h,
    },
    "NSS + SigLIP + LLaVA-feat": {
        "nss": X_nss, "siglip": X_siglip, "llava_feat": X_llava_feat,
    },
    "NSS + CLIP-H + LLaVA-feat": {
        "nss": X_nss, "clip_h": X_clip_h, "llava_feat": X_llava_feat,
    },

    # Full fusion (all streams)
    "FULL: NSS + SigLIP + CLIP-H + DINOv2 + LLaVA-feat": {
        "nss": X_nss, "siglip": X_siglip, "clip_h": X_clip_h,
        "dino": X_dino, "llava_feat": X_llava_feat,
    },
    "FULL+score: NSS + SigLIP + CLIP-H + DINOv2 + LLaVA-feat + LLaVA-score": {
        "nss": X_nss, "siglip": X_siglip, "clip_h": X_clip_h,
        "dino": X_dino, "llava_feat": X_llava_feat, "llava_score": X_llava_score,
    },
}

# Run
print("\n" + "=" * 100)
print(f"{'Model':<60} {'SROCC':>12} {'PLCC':>12} {'KROCC':>12}")
print("-" * 100)

results_log = []
best_srocc, best_label = -1, ""
for label, streams in variants.items():
    print(f"\n[{label}]", flush=True)
    sr_m, sr_s, pl_m, pl_s, kr_m, kr_s = evaluate_cv(streams, y, names, label)
    row = f"{label:<60} {sr_m:.4f}+/-{sr_s:.4f} {pl_m:.4f}+/-{pl_s:.4f} {kr_m:.4f}+/-{kr_s:.4f}"
    print(row, flush=True)
    results_log.append({
        "model": label,
        "srocc_mean": sr_m, "srocc_std": sr_s,
        "plcc_mean":  pl_m, "plcc_std":  pl_s,
        "krocc_mean": kr_m, "krocc_std": kr_s,
    })
    if sr_m > best_srocc:
        best_srocc, best_label = sr_m, label

print("\n" + "=" * 100)
print(f"Best: {best_label}  (SROCC={best_srocc:.4f})")

pd.DataFrame(results_log).to_csv("results/koniq_fusion_mlp_v3_results.csv", index=False)
print("\n-> saved summary: results/koniq_fusion_mlp_v3_results.csv")