"""
loss_ablation_liveitw.py - Compare 3 loss configurations on LIVE-itW

Same protocol as the main LIVE-itW experiment:
  - KonIQ pretraining
  - LIVE-itW fine-tuning (5-fold CV)
  - 3 ensemble seeds + 10 MC-dropout passes

Tests three loss configurations:
  1. MSE only
  2. MSE + PLCC
  3. MSE + PLCC + Rank (hybrid, used in main paper)

If the hybrid loss helps on small datasets, we expect to see a real gap here
because LIVE-itW has only ~930 training images per fold.

Runtime: ~3 hours on CPU (3 losses x 5 folds x 3 seeds x [pretrain 80ep + finetune 40ep])
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

# ===== Load KonIQ =====
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

X_koniq_nss    = np.array([koniq_nss[n]    for n in koniq_rows]).astype(np.float32)
X_koniq_siglip = np.array([koniq_siglip[n] for n in koniq_rows]).astype(np.float32)
X_koniq_clip_h = np.array([koniq_clip_h[n] for n in koniq_rows]).astype(np.float32)
y_koniq        = np.array([koniq_mos[n]    for n in koniq_rows]).astype(np.float32)
y_koniq_n      = (y_koniq - 1) * 25
print(f"  KonIQ: {len(koniq_rows)} samples")

# ===== Load LIVE-itW =====
print("Loading LIVE-itW...")
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
print(f"  LIVE-itW: {len(live_rows)} samples")

X_live_nss    = np.array([live_nss[n]    for n in live_rows]).astype(np.float32)
X_live_siglip = np.array([live_siglip[n] for n in live_rows]).astype(np.float32)
X_live_clip_h = np.array([live_clip_h[n] for n in live_rows]).astype(np.float32)
y_live        = np.array([live_mos[n]    for n in live_rows]).astype(np.float32)


# ===== Model =====
class MLPHead(nn.Module):
    def __init__(self, in_dim, h1=512, h2=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def plcc_loss(yp, yt, eps=1e-8):
    yp_c = yp - yp.mean(); yt_c = yt - yt.mean()
    num = (yp_c * yt_c).sum()
    den = torch.sqrt((yp_c**2).sum() * (yt_c**2).sum() + eps)
    return 1.0 - num / den

def ranking_loss(yp, yt):
    if yp.shape[0] < 2: return torch.tensor(0.0, device=yp.device)
    pd_ = yp.unsqueeze(0) - yp.unsqueeze(1)
    td  = yt.unsqueeze(0) - yt.unsqueeze(1)
    mask = (td.abs() > 1e-3).float()
    sign = torch.sign(td)
    loss = torch.nn.functional.softplus(-sign * pd_) * mask
    return loss.sum() / (mask.sum() + 1e-8)

def loss_mse_only(yp, yt):
    return nn.functional.mse_loss(yp, yt)
def loss_mse_plcc(yp, yt):
    return nn.functional.mse_loss(yp, yt) + 0.5 * plcc_loss(yp, yt)
def loss_hybrid(yp, yt):
    return (nn.functional.mse_loss(yp, yt)
            + 0.5 * plcc_loss(yp, yt)
            + 0.5 * ranking_loss(yp, yt))

LOSS_CONFIGS = [
    ("MSE only",          loss_mse_only),
    ("MSE + PLCC",        loss_mse_plcc),
    ("MSE + PLCC + Rank", loss_hybrid),
]

def preprocess(streams_pre, streams_ft, streams_te,
               pca_thresh=1000, pca_comp=256):
    parts_pre, parts_ft, parts_te = [], [], []
    for name in streams_pre:
        Xp, Xf, Xt = streams_pre[name], streams_ft[name], streams_te[name]
        Xall = np.concatenate([Xp, Xf], axis=0)
        sc = StandardScaler(); sc.fit(Xall)
        Xp = sc.transform(Xp); Xf = sc.transform(Xf); Xt = sc.transform(Xt)
        if Xp.shape[1] > pca_thresh:
            n_comp = min(pca_comp, Xp.shape[1], Xall.shape[0]-1)
            pca = PCA(n_components=n_comp, whiten=True, random_state=42)
            pca.fit(np.concatenate([Xp, Xf], axis=0))
            Xp = pca.transform(Xp); Xf = pca.transform(Xf); Xt = pca.transform(Xt)
        parts_pre.append(Xp.astype(np.float32))
        parts_ft.append(Xf.astype(np.float32))
        parts_te.append(Xt.astype(np.float32))
    return (np.concatenate(parts_pre, axis=1),
            np.concatenate(parts_ft, axis=1),
            np.concatenate(parts_te, axis=1))

def train_pretrain(X, y, loss_fn, epochs=80, lr=5e-4, bs=128, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPHead(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    n = Xt.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model

def finetune(model, X, y, loss_fn, epochs=40, lr=5e-5, bs=32, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    n = Xt.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model

def predict_tta(model, X, T=10):
    Xt = torch.from_numpy(X).to(device)
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(model(Xt).cpu().numpy())
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

ENSEMBLE_SEEDS = [42, 123, 456]
N_FOLDS = 5

print("\n" + "=" * 80)
print("Loss ablation on LIVE-itW (pretrain+finetune+ensemble pipeline)")
print(f"{'Loss config':<25} {'SROCC':>14} {'PLCC':>14} {'KROCC':>14}")
print("-" * 80)

all_results = []
all_preds   = {}

for loss_name, loss_fn in LOSS_CONFIGS:
    print(f"\n[{loss_name}]", flush=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_sroccs, fold_plccs, fold_kroccs = [], [], []
    all_oof_preds = np.zeros_like(y_live)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y_live), 1):
        streams_ft_train = {n: streams_full_live[n][train_idx] for n in ["nss","siglip","clip_h"]}
        streams_ft_test  = {n: streams_full_live[n][test_idx]  for n in ["nss","siglip","clip_h"]}

        X_pre, X_ft, X_te = preprocess(streams_full_pretrain, streams_ft_train, streams_ft_test)
        y_ft = y_live[train_idx]
        y_te = y_live[test_idx]

        ensemble_preds = []
        for seed in ENSEMBLE_SEEDS:
            model = train_pretrain(X_pre, y_koniq_n, loss_fn, seed=seed)
            model = finetune(model, X_ft, y_ft, loss_fn, seed=seed)
            preds = predict_tta(model, X_te, T=10)
            ensemble_preds.append(preds)

        y_pred = np.mean(ensemble_preds, axis=0)
        all_oof_preds[test_idx] = y_pred

        sr = stats.spearmanr(y_te,  y_pred).statistic
        pl = stats.pearsonr(y_te,   y_pred)[0]
        kr = stats.kendalltau(y_te, y_pred).statistic
        fold_sroccs.append(sr); fold_plccs.append(pl); fold_kroccs.append(kr)
        print(f"    fold {fold_i}/{N_FOLDS}  SROCC={sr:.4f}  PLCC={pl:.4f}", flush=True)

    sr_m, sr_s = np.mean(fold_sroccs), np.std(fold_sroccs)
    pl_m, pl_s = np.mean(fold_plccs),  np.std(fold_plccs)
    kr_m, kr_s = np.mean(fold_kroccs), np.std(fold_kroccs)
    print(f"  -> {loss_name}: SROCC={sr_m:.4f}+/-{sr_s:.4f}  PLCC={pl_m:.4f}+/-{pl_s:.4f}", flush=True)

    safe = loss_name.replace(" ", "_").replace("+", "and")
    csv_path = f"results/predictions/liveitw_loss_{safe}.csv"
    pd.DataFrame({"image_name": live_rows, "true_mos": y_live,
                  "pred_mos": all_oof_preds}).to_csv(csv_path, index=False)

    all_results.append({"loss": loss_name,
                        "srocc_mean": sr_m, "srocc_std": sr_s,
                        "plcc_mean":  pl_m, "plcc_std":  pl_s,
                        "krocc_mean": kr_m, "krocc_std": kr_s})
    all_preds[loss_name] = all_oof_preds.copy()

pd.DataFrame(all_results).to_csv("results/loss_ablation_liveitw_results.csv", index=False)
print("\n-> saved summary: results/loss_ablation_liveitw_results.csv")

print("\n" + "=" * 80)
print("Summary table:")
print(pd.DataFrame(all_results).to_string(index=False))
