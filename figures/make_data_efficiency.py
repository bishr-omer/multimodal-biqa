"""
fig8_data_efficiency.py - Generate Fig. 8: SROCC vs LIVE-itW training set size

Subsamples the LIVE-itW training fold to 20%, 40%, 60%, 80%, 100%
and evaluates the full pretrain+finetune+ensemble pipeline.

Single fold only (uses fold 0 of the 5-fold split) to keep runtime tractable.
Estimated runtime: ~30 min on CPU.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import scipy.io
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs("results/figures", exist_ok=True)

# Load KonIQ
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

# Load LIVE-itW
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

X_live_nss    = np.array([live_nss[n]    for n in live_rows]).astype(np.float32)
X_live_siglip = np.array([live_siglip[n] for n in live_rows]).astype(np.float32)
X_live_clip_h = np.array([live_clip_h[n] for n in live_rows]).astype(np.float32)
y_live        = np.array([live_mos[n]    for n in live_rows]).astype(np.float32)


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
    pd = yp.unsqueeze(0) - yp.unsqueeze(1)
    td = yt.unsqueeze(0) - yt.unsqueeze(1)
    mask = (td.abs() > 1e-3).float()
    sign = torch.sign(td)
    loss = torch.nn.functional.softplus(-sign * pd) * mask
    return loss.sum() / (mask.sum() + 1e-8)

def hybrid_loss(yp, yt):
    return (nn.functional.mse_loss(yp, yt)
            + 0.5 * plcc_loss(yp, yt)
            + 0.5 * ranking_loss(yp, yt))


def preprocess(streams_pre, streams_ft, streams_te,
               pca_thresh=1000, pca_comp=256):
    parts_pre, parts_ft, parts_te = [], [], []
    for name in streams_pre:
        Xp = streams_pre[name]
        Xf = streams_ft[name]
        Xt = streams_te[name]
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


def train_pretrain(X, y, epochs=80, lr=5e-4, bs=128, seed=42):
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
            loss = hybrid_loss(pred, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    return model

def finetune(model, X, y, epochs=40, lr=5e-5, bs=32, seed=42):
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
            loss = hybrid_loss(pred, yt[idx])
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


# Use fold 0 from the same KFold split as the main experiment
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(y_live))
train_idx, test_idx = splits[0]

print(f"\nFold 0: train={len(train_idx)}, test={len(test_idx)}")

# Streams (full three-stream)
streams_pre = {
    "nss":    X_koniq_nss,
    "siglip": X_koniq_siglip,
    "clip_h": X_koniq_clip_h,
}
streams_live_train = {
    "nss":    X_live_nss[train_idx],
    "siglip": X_live_siglip[train_idx],
    "clip_h": X_live_clip_h[train_idx],
}
streams_live_test = {
    "nss":    X_live_nss[test_idx],
    "siglip": X_live_siglip[test_idx],
    "clip_h": X_live_clip_h[test_idx],
}
y_train_full = y_live[train_idx]
y_test       = y_live[test_idx]

# Subsample percentages
percentages = [20, 40, 60, 80, 100]
ENSEMBLE_SEEDS = [42, 123, 456]

results = []
rng = np.random.RandomState(42)

for pct in percentages:
    n_keep = int(len(train_idx) * pct / 100)
    sub_idx = rng.choice(len(train_idx), size=n_keep, replace=False)
    sub_idx = np.sort(sub_idx)

    s_streams_train = {k: v[sub_idx] for k, v in streams_live_train.items()}
    sub_y_train = y_train_full[sub_idx]

    print(f"\n[{pct}%] training samples: {n_keep}")

    # Preprocess
    X_pre, X_ft, X_te = preprocess(streams_pre, s_streams_train, streams_live_test)

    # Ensemble
    ensemble_preds = []
    for seed in ENSEMBLE_SEEDS:
        model = train_pretrain(X_pre, y_koniq_n, seed=seed)
        model = finetune(model, X_ft, sub_y_train, seed=seed)
        preds = predict_tta(model, X_te, T=10)
        ensemble_preds.append(preds)
    y_pred = np.mean(ensemble_preds, axis=0)

    sr = stats.spearmanr(y_test, y_pred).statistic
    pl = stats.pearsonr(y_test,  y_pred)[0]

    print(f"  SROCC={sr:.4f}, PLCC={pl:.4f}")
    results.append({
        "percentage":   pct,
        "n_samples":    n_keep,
        "srocc":        sr,
        "plcc":         pl,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results/fig8_data_efficiency.csv", index=False)
print("\n-> saved CSV: results/fig8_data_efficiency.csv")
print(results_df)


# =============================================================
# Plot Figure 8
# =============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(results_df['percentage'], results_df['srocc'],
        marker='o', markersize=10, linewidth=2.5, color='#1a9850', label='SROCC')
ax.plot(results_df['percentage'], results_df['plcc'],
        marker='s', markersize=10, linewidth=2.5, color='#4575b4', label='PLCC')

# Annotate values
for _, r in results_df.iterrows():
    ax.annotate(f"{r['srocc']:.3f}",
                xy=(r['percentage'], r['srocc']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=8, color='#1a9850')
    ax.annotate(f"{r['plcc']:.3f}",
                xy=(r['percentage'], r['plcc']),
                xytext=(0, -16), textcoords='offset points',
                ha='center', fontsize=8, color='#4575b4')

ax.set_xlabel('LIVE-itW training set size (%)', fontsize=11)
ax.set_ylabel('Correlation coefficient', fontsize=11)
ax.set_title('Data efficiency on LIVE-itW (single fold, full pipeline)',
             fontsize=12, pad=10)
ax.set_xticks(percentages)
ax.set_xticklabels([f'{p}%\n({int(len(train_idx)*p/100)} imgs)' for p in percentages])
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('results/figures/fig8_data_efficiency.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig8_data_efficiency.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\n-> saved fig8_data_efficiency.{pdf,png}")
