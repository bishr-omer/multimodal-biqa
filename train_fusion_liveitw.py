"""
fusion_liveitw_finetune_mlp_v3.py - LIVE-itW with all three improvements:
1. Pretrain on KonIQ
2. Fine-tune on LIVE-itW (5-fold CV)
3. Test-time augmentation (MC dropout)
4. Ensemble 3 fine-tuning runs with different seeds

REDUCED SCOPE: 3 seeds, 4 variants (~3 hours total)
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

device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs("results/predictions", exist_ok=True)

# ===== Load KonIQ (pretrain) =====
print("Loading KonIQ...")
def load_dict(path):
    return np.load(path, allow_pickle=True).item()

koniq_nss_d    = load_dict("results/koniq_mvg_features.npy")
koniq_siglip_d = load_dict("results/koniq_siglip_features.npy")
koniq_clip_h_d = load_dict("results/koniq_clip_h14_features.npy")
koniq_df       = pd.read_csv("data/koniq10k/koniq10k_scores.csv")

koniq_nss    = dict(zip(koniq_nss_d["names"],    koniq_nss_d["features"]))
koniq_siglip = dict(zip(koniq_siglip_d["names"], koniq_siglip_d["features"]))
koniq_clip_h = dict(zip(koniq_clip_h_d["names"], koniq_clip_h_d["features"]))
koniq_mos    = dict(zip(koniq_df["image_name"],  koniq_df["MOS"]))

koniq_common = set(koniq_nss) & set(koniq_siglip) & set(koniq_clip_h) & set(koniq_mos)
koniq_rows   = sorted(koniq_common)
print(f"  KonIQ: {len(koniq_rows)} aligned")

X_koniq_nss    = np.array([koniq_nss[n]    for n in koniq_rows]).astype(np.float32)
X_koniq_siglip = np.array([koniq_siglip[n] for n in koniq_rows]).astype(np.float32)
X_koniq_clip_h = np.array([koniq_clip_h[n] for n in koniq_rows]).astype(np.float32)
y_koniq        = np.array([koniq_mos[n]    for n in koniq_rows]).astype(np.float32)
y_koniq_n      = (y_koniq - 1) * 25  # 1-5 -> 0-100

# ===== Load LIVE-itW =====
print("\nLoading LIVE-itW...")
live_nss_d    = load_dict("results/liveitw_mvg_features.npy")
live_siglip_d = load_dict("results/liveitw_siglip_features.npy")
live_clip_h_d = load_dict("results/liveitw_clip_h14_features.npy")

mos_mat = scipy.io.loadmat("data/liveitw/AllMOS_release.mat")
img_mat = scipy.io.loadmat("data/liveitw/AllImages_release.mat")
liveitw_mos_arr   = mos_mat['AllMOS_release'].flatten()
liveitw_names_all = [str(img_mat['AllImages_release'][i][0][0]).strip()
                     for i in range(len(liveitw_mos_arr))]

live_nss    = dict(zip(live_nss_d["names"],    live_nss_d["features"]))
live_siglip = dict(zip(live_siglip_d["names"], live_siglip_d["features"]))
live_clip_h = dict(zip(live_clip_h_d["names"], live_clip_h_d["features"]))
live_mos    = dict(zip(liveitw_names_all,      liveitw_mos_arr))

live_common = set(live_nss) & set(live_siglip) & set(live_clip_h) & set(live_mos)
live_rows   = sorted(live_common)
print(f"  LIVE-itW: {len(live_rows)} aligned")

X_live_nss    = np.array([live_nss[n]    for n in live_rows]).astype(np.float32)
X_live_siglip = np.array([live_siglip[n] for n in live_rows]).astype(np.float32)
X_live_clip_h = np.array([live_clip_h[n] for n in live_rows]).astype(np.float32)
y_live        = np.array([live_mos[n]    for n in live_rows]).astype(np.float32)


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

def ranking_loss(y_pred, y_true):
    n = y_pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=y_pred.device)
    pred_diff = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)
    true_diff = y_true.unsqueeze(0) - y_true.unsqueeze(1)
    mask = (true_diff.abs() > 1e-3).float()
    sign = torch.sign(true_diff)
    loss = torch.nn.functional.softplus(-sign * pred_diff) * mask
    return loss.sum() / (mask.sum() + 1e-8)

def hybrid_loss(y_pred, y_true, w_mse=1.0, w_plcc=0.5, w_rank=0.5):
    mse = nn.functional.mse_loss(y_pred, y_true)
    pl  = plcc_loss(y_pred, y_true)
    rk  = ranking_loss(y_pred, y_true)
    return w_mse * mse + w_plcc * pl + w_rank * rk


def preprocess_three_step(streams_pretrain, streams_finetune_train,
                           streams_test, pca_threshold=1000, pca_components=256):
    pretrain_parts, finetune_parts, test_parts = [], [], []
    for name in streams_pretrain:
        X_pre = streams_pretrain[name]
        X_ft  = streams_finetune_train[name]
        X_te  = streams_test[name]

        X_combined = np.concatenate([X_pre, X_ft], axis=0)

        scaler = StandardScaler()
        scaler.fit(X_combined)
        X_pre = scaler.transform(X_pre)
        X_ft  = scaler.transform(X_ft)
        X_te  = scaler.transform(X_te)

        if X_pre.shape[1] > pca_threshold:
            n_comp = min(pca_components, X_pre.shape[1], X_combined.shape[0] - 1)
            pca = PCA(n_components=n_comp, whiten=True, random_state=42)
            pca.fit(np.concatenate([X_pre, X_ft], axis=0))
            X_pre = pca.transform(X_pre)
            X_ft  = pca.transform(X_ft)
            X_te  = pca.transform(X_te)

        pretrain_parts.append(X_pre.astype(np.float32))
        finetune_parts.append(X_ft.astype(np.float32))
        test_parts.append(X_te.astype(np.float32))

    return (np.concatenate(pretrain_parts, axis=1),
            np.concatenate(finetune_parts, axis=1),
            np.concatenate(test_parts, axis=1))


def train_pretrain(X_pre, y_pre, epochs=80, batch_size=128, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_pre.shape[1]
    model = MLPHead(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.from_numpy(X_pre).to(device)
    y_t = torch.from_numpy(y_pre.astype(np.float32)).to(device)

    n = X_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = hybrid_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


def finetune(model, X_ft, y_ft, epochs=40, batch_size=32, lr=5e-5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_t = torch.from_numpy(X_ft).to(device)
    y_t = torch.from_numpy(y_ft.astype(np.float32)).to(device)

    n = X_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = hybrid_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


def predict_with_tta(model, X_test, dropout_passes=10):
    """TTA via Monte Carlo dropout."""
    X_t = torch.from_numpy(X_test).to(device)
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(dropout_passes):
            preds.append(model(X_t).cpu().numpy())
    model.eval()
    return np.mean(preds, axis=0)


# ===== Main =====

streams_full_pretrain = {
    "nss":    X_koniq_nss,
    "siglip": X_koniq_siglip,
    "clip_h": X_koniq_clip_h,
}
streams_full_live = {
    "nss":    X_live_nss,
    "siglip": X_live_siglip,
    "clip_h": X_live_clip_h,
}

# Reduced to 4 most important variants
variants = {
    "SigLIP only":           ["siglip"],
    "CLIP-H only":           ["clip_h"],
    "SigLIP + CLIP-H":       ["siglip", "clip_h"],
    "NSS + SigLIP + CLIP-H": ["nss", "siglip", "clip_h"],
}

ENSEMBLE_SEEDS = [42, 123, 456]
N_FOLDS = 5

print("\n" + "=" * 100)
print(f"LIVE-itW: pretrain on KonIQ -> fine-tune 5-fold CV")
print(f"Ensemble: {len(ENSEMBLE_SEEDS)} seeds, TTA: 10 MC-dropout passes")
print(f"{'Model':<30} {'SROCC':>14} {'PLCC':>14} {'KROCC':>14}")
print("-" * 100)

results_log = []
best_srocc, best_label = -1, ""

for label, stream_names in variants.items():
    print(f"\n[{label}]", flush=True)

    streams_pre  = {n: streams_full_pretrain[n] for n in stream_names}
    streams_live = {n: streams_full_live[n]    for n in stream_names}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_sroccs, fold_plccs, fold_kroccs = [], [], []
    all_preds = np.zeros_like(y_live)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y_live), 1):
        streams_ft_train = {n: streams_live[n][train_idx] for n in stream_names}
        streams_ft_test  = {n: streams_live[n][test_idx]  for n in stream_names}

        X_pre, X_ft, X_te = preprocess_three_step(streams_pre, streams_ft_train, streams_ft_test)
        y_ft = y_live[train_idx]
        y_te = y_live[test_idx]

        ensemble_preds = []
        for seed in ENSEMBLE_SEEDS:
            model = train_pretrain(X_pre, y_koniq_n, seed=seed)
            model = finetune(model, X_ft, y_ft, seed=seed)
            preds = predict_with_tta(model, X_te, dropout_passes=10)
            ensemble_preds.append(preds)

        y_pred = np.mean(ensemble_preds, axis=0)
        all_preds[test_idx] = y_pred

        sr = stats.spearmanr(y_te,  y_pred).statistic
        pl = stats.pearsonr(y_te,   y_pred)[0]
        kr = stats.kendalltau(y_te, y_pred).statistic
        fold_sroccs.append(sr); fold_plccs.append(pl); fold_kroccs.append(kr)
        print(f"    fold {fold_i}/{N_FOLDS}  SROCC={sr:.4f}  PLCC={pl:.4f}", flush=True)

    sr_m, sr_s = np.mean(fold_sroccs), np.std(fold_sroccs)
    pl_m, pl_s = np.mean(fold_plccs),  np.std(fold_plccs)
    kr_m, kr_s = np.mean(fold_kroccs), np.std(fold_kroccs)

    row = f"{label:<30} {sr_m:.4f}+/-{sr_s:.4f} {pl_m:.4f}+/-{pl_s:.4f} {kr_m:.4f}+/-{kr_s:.4f}"
    print(row, flush=True)

    safe_label = label.replace(" ", "_").replace("+", "and").replace("/", "-")
    csv_path = f"results/predictions/liveitw_finetune_{safe_label}.csv"
    pd.DataFrame({
        "image_name": live_rows,
        "true_mos":   y_live,
        "pred_mos":   all_preds,
    }).to_csv(csv_path, index=False)

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
pd.DataFrame(results_log).to_csv("results/liveitw_finetune_results.csv", index=False)
print("\n-> saved summary: results/liveitw_finetune_results.csv")
