"""
fig_gate_analysis.py - Visualize learned gates per distortion type on KADID

Two subplots:
  Left:  NSS gate values per distortion type (sorted by gate value)
         with delta SROCC from static fusion overlaid as color
  Right: All three gate values per distortion type as grouped curves
         with distortion category annotations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'font.size':   10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ===== Load data =====
gates_df = pd.read_csv('results/predictions/kadid10k_mult_per_distortion_gates.csv')
nss_contrib = pd.read_csv('results/kadid_nss_contribution_per_distortion.csv')

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

# Distortion categories for annotation
CATEGORIES = {
    "Blur":        [1, 2, 3],
    "Color":       [4, 5, 6, 7, 8],
    "Compression": [9, 10],
    "Noise":       [11, 12, 13, 14, 15],
    "Brightness":  [16, 17, 18, 19],
    "Spatial":     [20, 21, 22, 23, 24, 25],
}
CATEGORY_COLORS = {
    "Blur":        '#4575b4',
    "Color":       '#d73027',
    "Compression": '#fc8d59',
    "Noise":       '#fee090',
    "Brightness":  '#91bfdb',
    "Spatial":     '#1a9850',
}

def get_category(dist_type):
    for cat, types in CATEGORIES.items():
        if dist_type in types:
            return cat
    return "Other"

gates_df['category'] = gates_df['distortion_type'].apply(get_category)
gates_df['name'] = gates_df['distortion_type'].apply(
    lambda x: DISTORTION_NAMES.get(int(x), f"Type {x}"))

# Merge with NSS contribution data
merged = pd.merge(gates_df, nss_contrib[['distortion_type', 'delta_srocc']],
                  on='distortion_type', how='left')


# =============================================================
# FIGURE: 2 subplots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))


# ----- Left subplot: NSS gate per distortion, sorted -----
ax = axes[0]

# Sort by NSS gate value ascending
merged_sorted = merged.sort_values('gate_nss', ascending=True).reset_index(drop=True)

y_pos = np.arange(len(merged_sorted))

# Color bars by category
bar_colors = [CATEGORY_COLORS[get_category(int(r['distortion_type']))]
              for _, r in merged_sorted.iterrows()]

bars = ax.barh(y_pos, merged_sorted['gate_nss'], color=bar_colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)

ax.axvline(x=1.0, color='black', linewidth=1.2, linestyle='--',
           label='Neutral gate = 1.0', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{int(r['distortion_type']):>2}: {r['name']}"
                    for _, r in merged_sorted.iterrows()], fontsize=8)
ax.set_xlabel('Learned NSS gate weight', fontsize=11)
ax.set_title('NSS gate per distortion type\n(< 1.0 = suppressed, > 1.0 = amplified)',
             fontsize=11, pad=10)
ax.set_xlim(0.3, 1.4)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# Annotate with delta SROCC
for i, (_, row) in enumerate(merged_sorted.iterrows()):
    delta = row.get('delta_srocc', 0)
    if pd.notna(delta):
        symbol = f'+{delta:.3f}' if delta > 0 else f'{delta:.3f}'
        color  = '#1a9850' if delta > 0 else '#d73027'
        ax.text(row['gate_nss'] + 0.01, i, symbol,
                va='center', fontsize=7.5, color=color, fontweight='bold')

# Category legend
legend_patches = [mpatches.Patch(color=c, label=cat, alpha=0.85)
                  for cat, c in CATEGORY_COLORS.items()]
legend_patches.append(mpatches.Patch(color='white', label='─ ─ neutral gate'))
ax.legend(handles=legend_patches, loc='lower right', frameon=True,
          fontsize=8, title='Distortion category')


# ----- Right subplot: All 3 gate values as curves per category -----
ax = axes[1]

# Order distortion types by category for cleaner x-axis
ordered_types = []
cat_boundaries = []
cat_labels_positions = []
for cat, types in CATEGORIES.items():
    start = len(ordered_types)
    ordered_types.extend(types)
    end = len(ordered_types)
    cat_labels_positions.append((start + end) / 2)
    cat_boundaries.append(end)

# Build ordered gate arrays
gates_ordered = merged.set_index('distortion_type')
nss_vals    = [float(gates_ordered.loc[dt, 'gate_nss'])    for dt in ordered_types]
siglip_vals = [float(gates_ordered.loc[dt, 'gate_siglip']) for dt in ordered_types]
clip_h_vals = [float(gates_ordered.loc[dt, 'gate_clip_h']) for dt in ordered_types]

x = np.arange(len(ordered_types))

ax.plot(x, nss_vals,    marker='o', markersize=7, linewidth=2,
        color='#d73027', label='NSS gate')
ax.plot(x, siglip_vals, marker='s', markersize=7, linewidth=2,
        color='#4575b4', label='SigLIP gate')
ax.plot(x, clip_h_vals, marker='^', markersize=7, linewidth=2,
        color='#1a9850', label='CLIP-H gate')

ax.axhline(y=1.0, color='black', linewidth=1.0, linestyle='--',
           alpha=0.6, label='Neutral = 1.0')

# Category background shading
cat_colors_light = {
    "Blur":        '#e8f0fb',
    "Color":       '#fbe8e8',
    "Compression": '#fef3ea',
    "Noise":       '#fefbe8',
    "Brightness":  '#e8f4fb',
    "Spatial":     '#e8f5ee',
}
prev = 0
for (cat, types), bound in zip(CATEGORIES.items(), cat_boundaries):
    ax.axvspan(prev - 0.5, bound - 0.5,
               alpha=0.35, color=cat_colors_light[cat], zorder=0)
    ax.text((prev + bound - 1) / 2, 1.45, cat,
            ha='center', va='center', fontsize=8,
            color=CATEGORY_COLORS[cat], fontweight='bold')
    if bound < len(ordered_types):
        ax.axvline(x=bound - 0.5, color='gray', linewidth=0.5,
                   linestyle=':', alpha=0.7)
    prev = bound

ax.set_xticks(x)
ax.set_xticklabels([str(dt) for dt in ordered_types], fontsize=8)
ax.set_xlabel('Distortion type index (grouped by category)', fontsize=11)
ax.set_ylabel('Learned gate weight', fontsize=11)
ax.set_title('All stream gate weights per distortion type\n(grouped by category)',
             fontsize=11, pad=10)
ax.set_ylim(0.3, 1.55)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=True, fontsize=9)


plt.suptitle('Distortion-aware gate analysis on KADID-10k\n'
             'Multiplicative gating: learned NSS, SigLIP, and CLIP-H weights per distortion',
             fontsize=13, y=1.02)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_gate_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig_gate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"-> saved fig_gate_analysis.{{pdf,png}}")

# ===== Print key insights =====
print("\n=== Key insights from gate analysis ===")
merged_sorted_nss = merged.sort_values('gate_nss')
print("\nLowest NSS gates (most suppressed):")
for _, row in merged_sorted_nss.head(5).iterrows():
    delta = row.get('delta_srocc', float('nan'))
    print(f"  Type {int(row['distortion_type']):>2}: {row['name']:<25} gate={row['gate_nss']:.3f}  delta_SROCC={delta:+.4f}")

print("\nHighest NSS gates (most amplified):")
for _, row in merged_sorted_nss.tail(5).iterrows():
    delta = row.get('delta_srocc', float('nan'))
    print(f"  Type {int(row['distortion_type']):>2}: {row['name']:<25} gate={row['gate_nss']:.3f}  delta_SROCC={delta:+.4f}")

sr = stats.spearmanr(merged['gate_nss'], merged['delta_srocc']).statistic
print(f"\nSpearman correlation between NSS gate and delta SROCC: {sr:.4f}")
print("(Positive = model learned to amplify NSS where it helps and suppress where it hurts)")
