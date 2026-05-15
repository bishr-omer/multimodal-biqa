"""
loss_ablation_experiment.py - Compare 3 loss configurations on KonIQ-10k

Trains the full NSS + SigLIP + CLIP-H model with three different losses:
  1. MSE only
  2. MSE + PLCC
  3. MSE + PLCC + ranking (the hybrid loss used in the main paper)

Uses 5-fold CV for stability. Output:
  - results/loss_ablation_results.csv (summary metrics)
  - results/predictions/koniq_loss_*.csv (per-image OOF predictions)
  - results/figures/fig11_loss_ablation_scatter.{pdf,png}
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
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")

os.makedirs("results/predictions", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# ===== Load KonIQ features =====
def load_dict(path):
    return np.load(path, allow_pickle=True).item()

print("Loading KonIQ features...")
nss_data    = load_dict("results/koniq_mvg_features.npy")
siglip_data = load_dict("results/koniq_siglip_features.npy")
clip_h_data = load_dict("results/koniq_clip_h14_features.npy")
df          = pd.read_csv("data/koniq10k/koniq10k_scores.csv")

nss_dict    = dict(zip(nss_data["names"],    nss_data["features"]))
siglip_dict = dict(zip(siglip_data["names"], siglip_data["features"]))
clip_h_dict = dict(zip(clip_h_data["names"], clip_h_data["features"]))
mos_dict    = dict(zip(df["image_name"],     df["MOS"]))

common = set(nss_dict) & set(siglip_dict) & set(clip_h_dict) & set(mos_dict)
rows = sorted(common)
print(f"Aligned samples: {len(rows)}")

X_nss    = np.array([nss_dict[n]    for n in rows]).astype(np.float32)
X_siglip = np.array([siglip_dict[n] for n in rows]).astype(np.float32)
X_clip_h = np.array([clip_h_dict[n] for n in rows]).astype(np.float32)
y = np.array([mos_dict[n] for n in rows]).astype(np.float32)
names = np.array(rows)


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


# Three loss configurations
def loss_mse_only(yp, yt):
    return nn.functional.mse_loss(yp, yt)

def loss_mse_plcc(yp, yt):
    return nn.functional.mse_loss(yp, yt) + 0.5 * plcc_loss(yp, yt)

def loss_hybrid(yp, yt):
    return (nn.functional.mse_loss(yp, yt)
            + 0.5 * plcc_loss(yp, yt)
            + 0.5 * ranking_loss(yp, yt))


LOSS_CONFIGS = [
    ("MSE only",          loss_mse_only,  "#d73027"),
    ("MSE + PLCC",        loss_mse_plcc,  "#fdae61"),
    ("MSE + PLCC + Rank", loss_hybrid,    "#1a9850"),
]


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


def train_mlp(X_train, y_train, X_test, loss_fn,
              epochs=80, batch_size=128, lr=5e-4, weight_decay=1e-4):
    in_dim = X_train.shape[1]
    model = MLPHead(in_dim).to(device)
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
            opt.zero_grad()
            pred = model(X_tr_t[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        return model(X_te_t).cpu().numpy()


streams = {"nss": X_nss, "siglip": X_siglip, "clip_h": X_clip_h}

print("\n" + "=" * 80)
print("Loss ablation on KonIQ-10k, 5-fold CV, NSS + SigLIP + CLIP-H")
print(f"{'Loss config':<25} {'SROCC':>10} {'PLCC':>10} {'KROCC':>10}")
print("-" * 80)

all_results = []
all_preds   = {}

for loss_name, loss_fn, _color in LOSS_CONFIGS:
    print(f"\n[{loss_name}]", flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    sroccs, plccs, kroccs = [], [], []
    oof_preds = np.zeros_like(y)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y), 1):
        X_train, X_test = preprocess_streams(streams, train_idx, test_idx)
        y_pred = train_mlp(X_train, y[train_idx], X_test, loss_fn)
        oof_preds[test_idx] = y_pred

        y_true = y[test_idx]
        sr = stats.spearmanr(y_true, y_pred).statistic
        pl = stats.pearsonr(y_true, y_pred)[0]
        kr = stats.kendalltau(y_true, y_pred).statistic
        sroccs.append(sr); plccs.append(pl); kroccs.append(kr)
        print(f"    fold {fold_i}/5  SROCC={sr:.4f}  PLCC={pl:.4f}", flush=True)

    sr_m, sr_s = np.mean(sroccs), np.std(sroccs)
    pl_m, pl_s = np.mean(plccs),  np.std(plccs)
    kr_m, kr_s = np.mean(kroccs), np.std(kroccs)
    print(f"  -> {loss_name}: SROCC={sr_m:.4f}+/-{sr_s:.4f}  PLCC={pl_m:.4f}+/-{pl_s:.4f}", flush=True)

    safe = loss_name.replace(" ", "_").replace("+", "and")
    csv_path = f"results/predictions/koniq_loss_{safe}.csv"
    pd.DataFrame({"image_name": names, "true_mos": y,
                  "pred_mos": oof_preds}).to_csv(csv_path, index=False)

    all_results.append({"loss": loss_name,
                        "srocc_mean": sr_m, "srocc_std": sr_s,
                        "plcc_mean":  pl_m, "plcc_std":  pl_s,
                        "krocc_mean": kr_m, "krocc_std": kr_s})
    all_preds[loss_name] = oof_preds.copy()

pd.DataFrame(all_results).to_csv("results/loss_ablation_results.csv", index=False)
print("\n-> saved summary: results/loss_ablation_results.csv")


# ===== Figure: 3 scatter plots side by side =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   10,
    'axes.labelsize':  11,
    'axes.titlesize':  11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (loss_name, _fn, color), result in zip(axes, LOSS_CONFIGS, all_results):
    y_true = y
    y_pred = all_preds[loss_name]

    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color=color,
               edgecolors='none', rasterized=True)

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            'k--', linewidth=1.0, alpha=0.6, label='y = x')

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, p(xs), '-', color='black', linewidth=1.2, alpha=0.8,
            label='Linear fit')

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Ground-truth MOS', fontsize=11)
    ax.set_ylabel('Predicted MOS', fontsize=11)
    ax.set_title(f'{loss_name}\nSROCC = {result["srocc_mean"]:.4f}, '
                 f'PLCC = {result["plcc_mean"]:.4f}',
                 fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.set_aspect('equal', 'box')

plt.suptitle('Effect of loss function on prediction alignment (KonIQ-10k, 5-fold OOF)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig11_loss_ablation_scatter.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/fig11_loss_ablation_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> saved fig11_loss_ablation_scatter.{pdf,png}")

print("\n" + "=" * 80)
print("Summary table:")
print(pd.DataFrame(all_results).to_string(index=False))
