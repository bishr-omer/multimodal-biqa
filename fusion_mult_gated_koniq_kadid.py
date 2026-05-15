"""
fusion_mult_gated_koniq_kadid.py - Multiplicative gating on concatenated features

Variant 2 of distortion-aware fusion. Instead of mixing in low-dim space:

  1. Concatenate all 3 streams as in static fusion (650-dim total)
  2. Gating network produces 3 scalar weights (sigmoid x 2, range [0, 2])
  3. Each scalar multiplies its corresponding stream's contribution before concat
  4. MLP head operates on the full 650-dim weighted concat

This preserves the full information capacity of static fusion while allowing
input-conditioned scaling.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs("results/predictions", exist_ok=True)


# ===== Multiplicative-Gated Fusion Model =====
class MultGatedFusionModel(nn.Module):
    """
    For each input image x:
      1. NSS, SigLIP, CLIP-H features are concatenated (650-dim)
      2. Gating network outputs 3 scalar weights w_i in [0, 2]
      3. Each stream is scaled by its weight before concatenation
      4. Weighted concat -> 3-layer MLP -> MOS
    """
    def __init__(self, nss_dim, siglip_dim, clip_h_dim,
                 gate_hidden=64, head_hidden1=512, head_hidden2=256, dropout=0.3,
                 gate_max=2.0):
        super().__init__()

        self.nss_dim    = nss_dim
        self.siglip_dim = siglip_dim
        self.clip_h_dim = clip_h_dim
        self.gate_max   = gate_max

        # Gating network: takes SigLIP features as summary
        self.gate = nn.Sequential(
            nn.Linear(siglip_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )

        total_dim = nss_dim + siglip_dim + clip_h_dim

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(total_dim, head_hidden1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden1, head_hidden2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden2, 1),
        )

    def forward(self, x_nss, x_siglip, x_clip_h, return_gates=False):
        # Gates: sigmoid * gate_max -> [0, gate_max]
        gate_logits = self.gate(x_siglip)
        gate_weights = torch.sigmoid(gate_logits) * self.gate_max  # (B, 3)

        w_nss    = gate_weights[:, 0:1]
        w_siglip = gate_weights[:, 1:2]
        w_clip_h = gate_weights[:, 2:3]

        # Scale each stream, then concatenate
        scaled_nss    = w_nss    * x_nss
        scaled_siglip = w_siglip * x_siglip
        scaled_clip_h = w_clip_h * x_clip_h

        fused = torch.cat([scaled_nss, scaled_siglip, scaled_clip_h], dim=-1)
        pred  = self.head(fused).squeeze(-1)

        if return_gates:
            return pred, gate_weights
        return pred


# ===== Loss =====
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

def hybrid_loss(yp, yt):
    return (nn.functional.mse_loss(yp, yt)
            + 0.5 * plcc_loss(yp, yt)
            + 0.5 * ranking_loss(yp, yt))


def preprocess_streams(X_dict_train, X_dict_test,
                       pca_threshold=1000, pca_components=256):
    train_out, test_out = {}, {}
    for name in X_dict_train:
        X_tr = X_dict_train[name]
        X_te = X_dict_test[name]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        if X_tr.shape[1] > pca_threshold:
            n_comp = min(pca_components, X_tr.shape[1], X_tr.shape[0] - 1)
            pca = PCA(n_components=n_comp, whiten=True, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        train_out[name] = X_tr.astype(np.float32)
        test_out[name]  = X_te.astype(np.float32)
    return train_out, test_out


def train_mult_gated(X_train_dict, y_train, X_test_dict,
                     epochs=80, batch_size=128, lr=5e-4, weight_decay=1e-4,
                     return_gates=False):
    nss_dim    = X_train_dict["nss"].shape[1]
    siglip_dim = X_train_dict["siglip"].shape[1]
    clip_h_dim = X_train_dict["clip_h"].shape[1]

    model = MultGatedFusionModel(nss_dim=nss_dim, siglip_dim=siglip_dim,
                                  clip_h_dim=clip_h_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    x_nss_tr    = torch.from_numpy(X_train_dict["nss"]).to(device)
    x_siglip_tr = torch.from_numpy(X_train_dict["siglip"]).to(device)
    x_clip_h_tr = torch.from_numpy(X_train_dict["clip_h"]).to(device)
    y_tr        = torch.from_numpy(y_train.astype(np.float32)).to(device)

    x_nss_te    = torch.from_numpy(X_test_dict["nss"]).to(device)
    x_siglip_te = torch.from_numpy(X_test_dict["siglip"]).to(device)
    x_clip_h_te = torch.from_numpy(X_test_dict["clip_h"]).to(device)

    n = x_nss_tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(x_nss_tr[idx], x_siglip_tr[idx], x_clip_h_tr[idx])
            loss = hybrid_loss(pred, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        if return_gates:
            y_pred, gates = model(x_nss_te, x_siglip_te, x_clip_h_te, return_gates=True)
            return y_pred.cpu().numpy(), gates.cpu().numpy()
        else:
            return model(x_nss_te, x_siglip_te, x_clip_h_te).cpu().numpy(), None


def run_dataset(dataset_name, nss_path, siglip_path, clip_h_path,
                mos_path, mos_col, name_col):
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    def load_dict(path):
        return np.load(path, allow_pickle=True).item()

    nss_d    = load_dict(nss_path)
    siglip_d = load_dict(siglip_path)
    clip_h_d = load_dict(clip_h_path)

    nss_dict    = dict(zip(nss_d["names"],    nss_d["features"]))
    siglip_dict = dict(zip(siglip_d["names"], siglip_d["features"]))
    clip_h_dict = dict(zip(clip_h_d["names"], clip_h_d["features"]))

    df = pd.read_csv(mos_path)
    mos_dict = dict(zip(df[name_col], df[mos_col]))

    common = set(nss_dict) & set(siglip_dict) & set(clip_h_dict) & set(mos_dict)
    rows = sorted(common)
    print(f"  Aligned samples: {len(rows)}")

    X_nss    = np.array([nss_dict[n]    for n in rows]).astype(np.float32)
    X_siglip = np.array([siglip_dict[n] for n in rows]).astype(np.float32)
    X_clip_h = np.array([clip_h_dict[n] for n in rows]).astype(np.float32)
    y = np.array([mos_dict[n] for n in rows]).astype(np.float32)
    names = np.array(rows)

    # KADID distortion parsing
    dist_types  = np.full(len(rows), -1, dtype=int)
    dist_levels = np.full(len(rows), -1, dtype=int)
    if "kadid" in dataset_name.lower():
        for i, name in enumerate(rows):
            try:
                base = name.replace('.png','').replace('.jpg','')
                parts = base.split('_')
                if len(parts) >= 3 and parts[0].startswith('I'):
                    dist_types[i]  = int(parts[1])
                    dist_levels[i] = int(parts[2])
            except:
                pass

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_sroccs, fold_plccs, fold_kroccs = [], [], []
    all_preds = np.zeros_like(y)
    all_gates = np.zeros((len(y), 3))

    print(f"\n  {'Fold':<6} {'SROCC':>10} {'PLCC':>10} {'KROCC':>10}")
    print("  " + "-" * 50)

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(y), 1):
        X_train_dict = {"nss":    X_nss[train_idx],
                        "siglip": X_siglip[train_idx],
                        "clip_h": X_clip_h[train_idx]}
        X_test_dict  = {"nss":    X_nss[test_idx],
                        "siglip": X_siglip[test_idx],
                        "clip_h": X_clip_h[test_idx]}

        X_train_p, X_test_p = preprocess_streams(X_train_dict, X_test_dict)
        y_pred, gates = train_mult_gated(X_train_p, y[train_idx], X_test_p,
                                          return_gates=True)

        all_preds[test_idx] = y_pred
        all_gates[test_idx] = gates

        y_true = y[test_idx]
        sr = stats.spearmanr(y_true, y_pred).statistic
        pl = stats.pearsonr(y_true,  y_pred)[0]
        kr = stats.kendalltau(y_true, y_pred).statistic
        fold_sroccs.append(sr); fold_plccs.append(pl); fold_kroccs.append(kr)
        print(f"  {fold_i:<6} {sr:>10.4f} {pl:>10.4f} {kr:>10.4f}", flush=True)

    sr_m, sr_s = np.mean(fold_sroccs), np.std(fold_sroccs)
    pl_m, pl_s = np.mean(fold_plccs),  np.std(fold_plccs)
    kr_m, kr_s = np.mean(fold_kroccs), np.std(fold_kroccs)
    print(f"  {'Mean':<6} {sr_m:>10.4f} {pl_m:>10.4f} {kr_m:>10.4f}")
    print(f"  {'Std':<6} {sr_s:>10.4f} {pl_s:>10.4f} {kr_s:>10.4f}")
    print(f"\n  Mean learned gates (range [0, 2]):")
    print(f"    NSS:    {all_gates[:, 0].mean():.4f}  (std {all_gates[:, 0].std():.4f})")
    print(f"    SigLIP: {all_gates[:, 1].mean():.4f}  (std {all_gates[:, 1].std():.4f})")
    print(f"    CLIP-H: {all_gates[:, 2].mean():.4f}  (std {all_gates[:, 2].std():.4f})")

    safe = dataset_name.lower().replace(" ", "_").replace("-", "")
    out_df = pd.DataFrame({
        "image_name":  names,
        "true_mos":    y,
        "pred_mos":    all_preds,
        "gate_nss":    all_gates[:, 0],
        "gate_siglip": all_gates[:, 1],
        "gate_clip_h": all_gates[:, 2],
    })
    if "kadid" in dataset_name.lower():
        out_df["distortion_type"]  = dist_types
        out_df["distortion_level"] = dist_levels
    out_df.to_csv(f"results/predictions/{safe}_mult_gated_fusion.csv", index=False)

    if "kadid" in dataset_name.lower():
        per_dist_gates = []
        for dt in range(1, 26):
            mask = dist_types == dt
            if mask.sum() < 10: continue
            per_dist_gates.append({
                "distortion_type": dt,
                "n_samples":       int(mask.sum()),
                "gate_nss":        float(all_gates[mask, 0].mean()),
                "gate_siglip":     float(all_gates[mask, 1].mean()),
                "gate_clip_h":     float(all_gates[mask, 2].mean()),
                "srocc":           float(stats.spearmanr(y[mask], all_preds[mask]).statistic),
            })
        pd.DataFrame(per_dist_gates).to_csv(
            f"results/predictions/{safe}_mult_per_distortion_gates.csv", index=False)
        print(f"  -> saved per-distortion gates: results/predictions/{safe}_mult_per_distortion_gates.csv")

    return {
        "dataset":    dataset_name,
        "srocc_mean": sr_m, "srocc_std": sr_s,
        "plcc_mean":  pl_m, "plcc_std":  pl_s,
        "krocc_mean": kr_m, "krocc_std": kr_s,
    }


# ===== Main =====
all_results = []

all_results.append(run_dataset(
    "KonIQ-10k",
    "results/koniq_mvg_features.npy",
    "results/koniq_siglip_features.npy",
    "results/koniq_clip_h14_features.npy",
    "data/koniq10k/koniq10k_scores.csv",
    mos_col="MOS", name_col="image_name",
))

all_results.append(run_dataset(
    "KADID-10k",
    "results/kadid_mvg_features.npy",
    "results/kadid_siglip_features.npy",
    "results/kadid_clip_h14_features.npy",
    "data/kadid10k/dmos.csv",
    mos_col="dmos", name_col="dist_img",
))

pd.DataFrame(all_results).to_csv("results/mult_gated_fusion_results.csv", index=False)
print("\n" + "=" * 80)
print("Summary:")
print(pd.DataFrame(all_results).to_string(index=False))
print("\n-> saved summary: results/mult_gated_fusion_results.csv")
print("\nCompare to static fusion (concat):")
print("  KonIQ:   0.9142 (target to beat)")
print("  KADID:   0.9715 (target to beat)")
