import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

# ── Load features ──────────────────────────────────────────────────────────────
clip_data  = np.load("results/koniq_clip_features.npy",  allow_pickle=True).item()
mvg_data   = np.load("results/koniq_mvg_features.npy",   allow_pickle=True).item()
llava_data = np.load("koniq_llava_features.npy",          allow_pickle=True).item()

# ── Load MOS scores ────────────────────────────────────────────────────────────
df = pd.read_csv("data/koniq10k/koniq10k_scores.csv").set_index("image_name")

# ── Build lookup dicts ─────────────────────────────────────────────────────────
clip_dict  = dict(zip(clip_data["names"],  clip_data["features"]))
mvg_dict   = dict(zip(mvg_data["names"],   mvg_data["features"]))
llava_feat_dict  = dict(zip(llava_data["names"], llava_data["features"]))
llava_score_dict = dict(zip(llava_data["names"], llava_data["scores"]))

# ── Align all streams ──────────────────────────────────────────────────────────
X_clip, X_mvg, X_llava_feat, X_llava_score, y = [], [], [], [], []

for name in clip_dict:
    if (name in mvg_dict
            and name in llava_feat_dict
            and name in df.index):
        X_clip.append(clip_dict[name])
        X_mvg.append(mvg_dict[name])
        X_llava_feat.append(llava_feat_dict[name])
        X_llava_score.append([llava_score_dict[name]])
        y.append(df.loc[name, "MOS"])

X_clip       = np.array(X_clip)        # (N, 512)
X_mvg        = np.array(X_mvg)         # (N, 138)
X_llava_feat = np.array(X_llava_feat)  # (N, 4096)
X_llava_score= np.array(X_llava_score) # (N, 1)
y            = np.array(y)

N = len(y)
print(f"Aligned samples  : {N}")
print(f"CLIP shape       : {X_clip.shape}")
print(f"MVG shape        : {X_mvg.shape}")
print(f"LLaVA feat shape : {X_llava_feat.shape}")
print(f"LLaVA score shape: {X_llava_score.shape}")

# ── Fusion variants ────────────────────────────────────────────────────────────
variants = {
    "CLIP only (baseline)"      : X_clip,
    "MVG only  (baseline)"      : X_mvg,
    "LLaVA score only"          : X_llava_score,
    "CLIP + MVG"                : np.concatenate([X_clip, X_mvg],                          axis=1),
    "CLIP + LLaVA score"        : np.concatenate([X_clip, X_llava_score],                  axis=1),
    "MVG  + LLaVA score"        : np.concatenate([X_mvg,  X_llava_score],                  axis=1),
    "CLIP + MVG + LLaVA score"  : np.concatenate([X_clip, X_mvg, X_llava_score],           axis=1),
    "CLIP + MVG + LLaVA feat"   : np.concatenate([X_clip, X_mvg, X_llava_feat],            axis=1),
    "CLIP + MVG + LLaVA full"   : np.concatenate([X_clip, X_mvg, X_llava_feat, X_llava_score], axis=1),
}

# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    srocc, _ = stats.spearmanr(y_test,  y_pred)
    plcc,  _ = stats.pearsonr(y_test,   y_pred)
    krocc, _ = stats.kendalltau(y_test, y_pred)
    return srocc, plcc, krocc

print("\n" + "="*62)
print(f"{'Model':<36} {'SROCC':>7} {'PLCC':>7} {'KROCC':>7}")
print("-"*62)

results = {}
for label, X in variants.items():
    srocc, plcc, krocc = evaluate(X, y, label)
    results[label] = (srocc, plcc, krocc)
    print(f"{label:<36} {srocc:>7.4f} {plcc:>7.4f} {krocc:>7.4f}")

print("="*62)

best_label = max(results, key=lambda k: results[k][0])
print(f"\nBest model by SROCC: {best_label}")
print(f"  SROCC={results[best_label][0]:.4f}  PLCC={results[best_label][1]:.4f}  KROCC={results[best_label][2]:.4f}")