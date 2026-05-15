"""
fusion_liveitw_koniq_only.py - LIVE-itW transfer trained ONLY on KonIQ
Both are authentic distortion datasets with similar MOS distributions.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.io
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs("results/predictions", exist_ok=True)

# ===== Load KonIQ (training) =====
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

# ===== Load LIVE-itW (test) =====
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

# Normalize KonIQ MOS to LIVE-itW range (0-100)
y_koniq_n = (y_koniq - 1) * 25  # 1-5 -> 0-100

print(f"\nMOS ranges:")
print(f"  KonIQ (train, normalized): [{y_koniq_n.min():.1f}, {y_koniq_n.max():.1f}]")
print(f"  LIVE-itW (test):           [{y_live.min():.1f}, {y_live.max():.1f}]")
print(f"\nTraining: {len(y_koniq_n)}")
print(f"Test:     {len(y_live)}")


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

def preprocess_streams_train_test(streams_train, streams_test,
                                   pca_threshold=1000, pca_components=256):
    train_parts, test_parts = [], []
    for name in streams_train:
        X_tr = streams_train[name]
        X_te = streams_test[name]
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

def train_and_predict(streams_train, streams_test, y_train,
                      epochs=80, batch_size=128, lr=5e-4):
    X_tr, X_te = preprocess_streams_train_test(streams_train, streams_test)
    in_dim = X_tr.shape[1]
    model = MLPHead(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)

    n = X_tr_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = hybrid_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1}/{epochs}", flush=True)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()
    return y_pred


# ===== Variants =====
variants = {
    "NSS only": {
        "train": {"nss": X_koniq_nss},
        "test":  {"nss": X_live_nss},
    },
    "SigLIP only": {
        "train": {"siglip": X_koniq_siglip},
        "test":  {"siglip": X_live_siglip},
    },
    "CLIP-H only": {
        "train": {"clip_h": X_koniq_clip_h},
        "test":  {"clip_h": X_live_clip_h},
    },
    "NSS + SigLIP": {
        "train": {"nss": X_koniq_nss, "siglip": X_koniq_siglip},
        "test":  {"nss": X_live_nss,  "siglip": X_live_siglip},
    },
    "NSS + CLIP-H": {
        "train": {"nss": X_koniq_nss, "clip_h": X_koniq_clip_h},
        "test":  {"nss": X_live_nss,  "clip_h": X_live_clip_h},
    },
    "SigLIP + CLIP-H": {
        "train": {"siglip": X_koniq_siglip, "clip_h": X_koniq_clip_h},
        "test":  {"siglip": X_live_siglip,  "clip_h": X_live_clip_h},
    },
    "NSS + SigLIP + CLIP-H": {
        "train": {"nss": X_koniq_nss, "siglip": X_koniq_siglip, "clip_h": X_koniq_clip_h},
        "test":  {"nss": X_live_nss,  "siglip": X_live_siglip,  "clip_h": X_live_clip_h},
    },
}

# ===== Run =====
print("\n" + "=" * 100)
print("Cross-dataset transfer: KonIQ -> LIVE-itW")
print(f"{'Model':<30} {'SROCC':>12} {'PLCC':>12} {'KROCC':>12}")
print("-" * 100)

results_log = []
for label, cfg in variants.items():
    print(f"\n[{label}]", flush=True)
    y_pred = train_and_predict(cfg["train"], cfg["test"], y_koniq_n)

    sr = stats.spearmanr(y_live,  y_pred).statistic
    pl = stats.pearsonr(y_live,   y_pred)[0]
    kr = stats.kendalltau(y_live, y_pred).statistic

    row = f"{label:<30} {sr:>12.4f} {pl:>12.4f} {kr:>12.4f}"
    print(row, flush=True)

    safe_label = label.replace(" ", "_").replace("+", "and").replace("/", "-")
    csv_path = f"results/predictions/liveitw_koniq_{safe_label}.csv"
    pd.DataFrame({
        "image_name": live_rows,
        "true_mos":   y_live,
        "pred_mos":   y_pred,
    }).to_csv(csv_path, index=False)

    results_log.append({
        "model": label,
        "srocc": sr, "plcc": pl, "krocc": kr,
    })

print("\n" + "=" * 100)
pd.DataFrame(results_log).to_csv("results/liveitw_koniq_only_results.csv", index=False)
print("-> saved summary: results/liveitw_koniq_only_results.csv")
