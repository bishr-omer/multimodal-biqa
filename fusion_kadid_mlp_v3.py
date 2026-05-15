"""
fusion_kadid_mlp_v3.py - KADID-10k MLP fusion with ranking loss
Streams: NSS (138) + SigLIP (1152) + CLIP-H (1024)
Regressor: 3-layer MLP, hybrid loss (MSE + PLCC + ranking)
5-fold CV, dumps per-image predictions to CSV
Includes per-distortion-type analysis (25 types x 5 levels)
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs("results/predictions", exist_ok=True)

# ===== Load features (dict-style local files) =====
def load_dict(path):
    return np.load(path, allow_pickle=True).item()

print("Loading KADID features...")
nss_data    = load_dict("results/kadid_mvg_features.npy")
siglip_data = load_dict("results/kadid_siglip_features.npy")
clip_h_data = load_dict("results/kadid_clip_h14_features.npy")

# Load MOS from CSV
df = pd.read_csv("data/kadid10k/dmos.csv")
print(f"KADID CSV columns: {df.columns.tolist()}")

# Build dicts
nss_dict    = dict(zip(nss_data["names"],    nss_data["features"]))
siglip_dict = dict(zip(siglip_data["names"], siglip_data["features"]))
clip_h_dict = dict(zip(clip_h_data["names"], clip_h_data["features"]))
mos_dict    = dict(zip(df['dist_img'], df['dmos']))

# Align
common = set(nss_dict) & set(siglip_dict) & set(clip_h_dict) & set(mos_dict)
rows = sorted(common)
print(f"Aligned samples: {len(rows)}")

X_nss    = np.array([nss_dict[n]    for n in rows]).astype(np.float32)
X_siglip = np.array([siglip_dict[n] for n in rows]).astype(np.float32)
X_clip_h = np.array([clip_h_dict[n] for n in rows]).astype(np.float32)
y = np.array([mos_dict[n] for n in rows]).astype(np.float32)
names = np.array(rows)

# Parse KADID filename: I{img_id:02d}_{dist_type:02d}_{level:02d}.png
def parse_kadid_name(name):
    try:
        base = name.replace('.png', '').replace('.jpg', '')
        parts = base.split('_')
        if len(parts) >= 3 and parts[0].startswith('I'):
            return int(parts[0][1:]), int(parts[1]), int(parts[2])
    except:
        pass
    return None

DISTORTION_NAMES = {
    1:  "Gaussian blur",         2:  "Lens blur",            3:  "Motion blur",
    4:  "Color diffusion",        5:  "Color shift",          6:  "Color quantization",
    7:  "Color saturation 1",     8:  "Color saturation 2",   9:  "JPEG2000",
    10: "JPEG",                  11: "White noise",          12: "White noise CC",
    13: "Impulse noise",          14: "Multiplicative noise", 15: "Denoise",
    16: "Brighten",              17: "Darken",               18: "Mean shift",
    19: "Jitter",                20: "Non-eccentricity patch",21: "Pixelate",
    22: "Quantization",           23: "Color block",          24: "High sharpen",
    25: "Contrast change",
}

parsed       = [parse_kadid_name(n) for n in rows]
dist_types   = np.array([p[1] if p else -1 for p in parsed])
dist_levels  = np.array([p[2] if p else -1 for p in parsed])
print(f"Parseable filenames: {(dist_types > 0).sum()}/{len(rows)}")

print(f"NSS:    {X_nss.shape}")
print(f"SigLIP: {X_siglip.shape}")
print(f"CLIP-H: {X_clip_h.shape}")

# ===== Model and loss =====
class MLPHead(nn.Module):
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
    yp = y_pred - y_pred.mean()
    yt = y_true - y_true.mean()
    num = (yp * yt).sum()
    den = torch.sqrt((yp ** 2).sum() * (yt ** 2).sum() + eps)
    return 1.0 - num / den

def ranking_loss(y_pred, y_true, margin=0.0):
    n = y_pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=y_pred.device)
    pred_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
    true_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)
    mask = (true_diff.abs() > 1e-3).float()
    sign = torch.sign(true_diff)
    loss = torch.nn.functional.softplus(-sign * pred_diff + margin) * mask
    return loss.sum() / (mask.sum() + 1e-8)

def hybrid_loss(y_pred, y_true, w_mse=1.0, w_plcc=0.5, w_rank=0.5):
    mse = nn.functional.mse_loss(y_pred, y_true)
    pl  = plcc_loss(y_pred, y_true)
    rk  = ranking_loss(y_pred, y_true)
    return w_mse * mse + w_plcc * pl + w_rank * rk, (mse.item(), pl.item(), rk.item())

def preprocess_streams(streams_dict, train_idx, test_idx,
                       pca_threshold=1000, pca_components=256):
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
              hidden1=512, hidden2=256, dropout=0.3):
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
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss, _ = hybrid_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    return y_pred

def evaluate_cv(streams_dict, y, names, label, n_splits=5, save_csv=True):
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
        csv_path = f"results/predictions/kadid_{safe_label}.csv"
        pd.DataFrame({
            "image_name":      names,
            "true_mos":        y,
            "pred_mos":        all_preds,
            "distortion_type": dist_types,
            "distortion_level":dist_levels,
        }).to_csv(csv_path, index=False)
        print(f"    -> saved per-image preds: {csv_path}")

    return (np.mean(sroccs), np.std(sroccs),
            np.mean(plccs),  np.std(plccs),
            np.mean(kroccs), np.std(kroccs),
            all_preds)


# ===== Variants =====
variants = {
    "NSS only":              {"nss": X_nss},
    "SigLIP only":           {"siglip": X_siglip},
    "CLIP-H only":           {"clip_h": X_clip_h},
    "NSS + SigLIP":          {"nss": X_nss, "siglip": X_siglip},
    "NSS + CLIP-H":          {"nss": X_nss, "clip_h": X_clip_h},
    "SigLIP + CLIP-H":       {"siglip": X_siglip, "clip_h": X_clip_h},
    "NSS + SigLIP + CLIP-H": {"nss": X_nss, "siglip": X_siglip, "clip_h": X_clip_h},
}

# ===== Run =====
print("\n" + "=" * 100)
print(f"{'Model':<40} {'SROCC':>14} {'PLCC':>14} {'KROCC':>14}")
print("-" * 100)

results_log = []
for label, streams in variants.items():
    print(f"\n[{label}]", flush=True)
    sr_m, sr_s, pl_m, pl_s, kr_m, kr_s, _ = evaluate_cv(streams, y, names, label)
    row = f"{label:<40} {sr_m:.4f}+/-{sr_s:.4f} {pl_m:.4f}+/-{pl_s:.4f} {kr_m:.4f}+/-{kr_s:.4f}"
    print(row, flush=True)
    results_log.append({
        "model": label,
        "srocc_mean": sr_m, "srocc_std": sr_s,
        "plcc_mean":  pl_m, "plcc_std":  pl_s,
        "krocc_mean": kr_m, "krocc_std": kr_s,
    })

pd.DataFrame(results_log).to_csv("results/kadid_fusion_mlp_v3_results.csv", index=False)
print("\n-> saved summary: results/kadid_fusion_mlp_v3_results.csv")

# ===== Per-distortion analysis =====
print("\n" + "=" * 100)
print("PER-DISTORTION-TYPE ANALYSIS (NSS + SigLIP + CLIP-H)")
print("=" * 100)

target_csv = "results/predictions/kadid_NSS_and_SigLIP_and_CLIP-H.csv"
if not os.path.exists(target_csv):
    print(f"WARNING: {target_csv} not found")
else:
    pred_df = pd.read_csv(target_csv)
    print(f"\n{'Type':<5} {'Distortion Name':<25} {'N':<6} {'SROCC':>8} {'PLCC':>8}")
    print("-" * 60)

    per_dist_results = []
    for dist_t in sorted(set(pred_df['distortion_type'].values)):
        if dist_t < 1 or dist_t > 25:
            continue
        sub = pred_df[pred_df['distortion_type'] == dist_t]
        if len(sub) < 10:
            continue
        sr = stats.spearmanr(sub['true_mos'], sub['pred_mos']).statistic
        pl = stats.pearsonr(sub['true_mos'], sub['pred_mos'])[0]
        name = DISTORTION_NAMES.get(int(dist_t), f"Type {dist_t}")
        print(f"{int(dist_t):<5} {name:<25} {len(sub):<6} {sr:>8.4f} {pl:>8.4f}")
        per_dist_results.append({
            "distortion_type": int(dist_t),
            "distortion_name": name,
            "n_samples":       len(sub),
            "srocc":           sr,
            "plcc":            pl,
        })
    pd.DataFrame(per_dist_results).to_csv("results/kadid_per_distortion_results.csv", index=False)
    print(f"\n-> saved per-distortion results: results/kadid_per_distortion_results.csv")

# ===== NSS contribution per distortion =====
nss_csv   = "results/predictions/kadid_NSS_and_SigLIP_and_CLIP-H.csv"
nonss_csv = "results/predictions/kadid_SigLIP_and_CLIP-H.csv"

if os.path.exists(nss_csv) and os.path.exists(nonss_csv):
    print("\n" + "=" * 100)
    print("NSS CONTRIBUTION PER DISTORTION TYPE")
    print("=" * 100)
    print(f"\n{'Type':<5} {'Distortion Name':<25} {'No NSS':>10} {'With NSS':>10} {'Delta':>10}")
    print("-" * 70)

    df_nss   = pd.read_csv(nss_csv)
    df_nonss = pd.read_csv(nonss_csv)

    nss_contrib = []
    for dist_t in sorted(set(df_nss['distortion_type'].values)):
        if dist_t < 1 or dist_t > 25:
            continue
        sub_nss   = df_nss[df_nss['distortion_type'] == dist_t]
        sub_nonss = df_nonss[df_nonss['distortion_type'] == dist_t]
        if len(sub_nss) < 10:
            continue
        sr_nss   = stats.spearmanr(sub_nss['true_mos'],   sub_nss['pred_mos']).statistic
        sr_nonss = stats.spearmanr(sub_nonss['true_mos'], sub_nonss['pred_mos']).statistic
        delta    = sr_nss - sr_nonss
        name = DISTORTION_NAMES.get(int(dist_t), f"Type {dist_t}")
        print(f"{int(dist_t):<5} {name:<25} {sr_nonss:>10.4f} {sr_nss:>10.4f} {delta:>+10.4f}")
        nss_contrib.append({
            "distortion_type":  int(dist_t),
            "distortion_name":  name,
            "srocc_no_nss":     sr_nonss,
            "srocc_with_nss":   sr_nss,
            "delta_srocc":      delta,
        })

    pd.DataFrame(nss_contrib).to_csv("results/kadid_nss_contribution_per_distortion.csv", index=False)
    print(f"\n-> saved NSS contribution: results/kadid_nss_contribution_per_distortion.csv")
