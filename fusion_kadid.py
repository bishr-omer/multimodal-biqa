import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

# ── Load features ──────────────────────────────────────────────────────────────
clip_data = np.load("results/kadid_clip_features.npy", allow_pickle=True).item()
mvg_data  = np.load("results/kadid_mvg_features.npy",  allow_pickle=True).item()

# ── Load DMOS scores ───────────────────────────────────────────────────────────
df = pd.read_csv("data/kadid10k/dmos.csv").set_index("dist_img")

# ── Build lookup dicts ─────────────────────────────────────────────────────────
clip_dict = dict(zip(clip_data["names"], clip_data["features"]))
mvg_dict  = dict(zip(mvg_data["names"],  mvg_data["features"]))

# ── Align ──────────────────────────────────────────────────────────────────────
X_clip, X_mvg, y = [], [], []

for name in clip_dict:
    if name in mvg_dict and name in df.index:
        X_clip.append(clip_dict[name])
        X_mvg.append(mvg_dict[name])
        y.append(df.loc[name, "dmos"])

X_clip = np.array(X_clip)
X_mvg  = np.array(X_mvg)
y      = np.array(y)

print(f"Aligned samples : {len(y)}")
print(f"CLIP shape      : {X_clip.shape}")
print(f"MVG shape       : {X_mvg.shape}")

# ── Fusion variants ────────────────────────────────────────────────────────────
variants = {
    "CLIP only (baseline)" : X_clip,
    "MVG only  (baseline)" : X_mvg,
    "CLIP + MVG (fusion)"  : np.concatenate([X_clip, X_mvg], axis=1),
}

# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(X, y):
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

print("\n" + "="*58)
print(f"{'Model':<32} {'SROCC':>7} {'PLCC':>7} {'KROCC':>7}")
print("-"*58)
for label, X in variants.items():
    srocc, plcc, krocc = evaluate(X, y)
    print(f"{label:<32} {srocc:>7.4f} {plcc:>7.4f} {krocc:>7.4f}")
print("="*58)
