"""
fusion_liveitw_intra_mlp_v3.py - LIVE-itW intra-dataset 80/20 split MLP fusion
Streams: NSS (138) + SigLIP (1152) + CLIP-H (1024)
5-fold CV on LIVE-itW alone (no cross-dataset transfer)
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
import scipy.io
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs("results/predictions", exist_ok=True)

# ===== Load LIVE-itW =====
print("Loading LIVE-itW features...")
def load_dict(path):
    return np.load(path, allow_pickle=True).item()

nss_d    = load_dict("results/liveitw_mvg_features.npy")
siglip_d = load_dict("results/liveitw_siglip_features.npy")
clip_h_d = load_dict("results/liveitw_clip_h14_features.npy")

mos_mat = scipy.io.loadmat("data/liveitw/AllMOS_release.mat")
img_mat = scipy.io.loadmat("data/liveitw/AllImages_release.mat")
mos_arr   = mos_mat['AllMOS_release'].flatten()
names_all = [str(img_mat['AllImages_release'][i][0][0]).strip()
             for i in range(len(mos_arr))]

nss_dict    = dict(zip(nss_d["names"],    nss_d["features"]))
siglip_dict = dict(zip(siglip_d["names"], siglip_d["features"]))
clip_h_dict = dict(zip(clip_h_d["names"], clip_h_d["features"]))
mos_dict    = dict(zip(names_all,         mos_arr))

common = set(nss_dict) & set(siglip_dict) & set(clip_h_dict) & set(mos_dict)
rows = sorted(common)
print(f"Aligned samples: {len(rows)}")

X_nss    = np.array([nss_dict[n]    for n in rows]).astype(np.float32)
X_siglip = np.array([siglip_dict[n] for n in rows]).astype(np.float32)
X_clip_h = np.array([clip_h_dict[n] for n in rows]).astype(np.float32)
y = np.array([mos_dict[n] for n in rows]).astype(np.float32)
names = np.array(rows)

print(f"NSS:    {X_nss.shape}")
print(f"SigLIP: {X_siglip.shape}")
print(f"CLIP-H: {X_clip_h.shape}")
print(f"MOS range: [{y.min():.2f}, {y.max():.2f}]")

# ===== Model =====
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
              epochs=80, batch_size=64, lr=5e-4, weight_decay=1e-4,
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
        csv_path = f"results/predictions/liveitw_intra_{safe_label}.csv"
        pd.DataFrame({
            "image_name": names,
            "true_mos":   y,
            "pred_mos":   all_preds,
        }).to_csv(csv_path, index=False)

    return (np.mean(sroccs), np.std(sroccs),
            np.mean(plccs),  np.std(plccs),
            np.mean(kroccs), np.std(kroccs))


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
print("LIVE-itW intra-dataset 5-fold CV")
print(f"{'Model':<40} {'SROCC':>14} {'PLCC':>14} {'KROCC':>14}")
print("-" * 100)

results_log = []
best_srocc, best_label = -1, ""
for label, streams in variants.items():
    print(f"\n[{label}]", flush=True)
    sr_m, sr_s, pl_m, pl_s, kr_m, kr_s = evaluate_cv(streams, y, names, label)
    row = f"{label:<40} {sr_m:.4f}+/-{sr_s:.4f} {pl_m:.4f}+/-{pl_s:.4f} {kr_m:.4f}+/-{kr_s:.4f}"
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

pd.DataFrame(results_log).to_csv("results/liveitw_intra_results.csv", index=False)
print("\n-> saved summary: results/liveitw_intra_results.csv")
