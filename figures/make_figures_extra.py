"""
make_paper_figures_v3.py - Additional figures for the paper

Fig. 6: Stream-pair prediction correlation heatmap (3x3 matrix, per dataset)
Fig. 9: Distortion type x severity SROCC heatmap (KADID, 25x5 grid)
Fig. 10: Qualitative examples grid (best/worst predictions across datasets)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
import scipy.io
import warnings
warnings.filterwarnings('ignore')

# ===== Style =====
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
})

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================
# FIGURE 6: Stream correlation heatmap
# =============================================================
# Pearson correlation between predictions from each individual stream model
# (NSS-only, SigLIP-only, CLIP-H-only). One subplot per dataset.

print("Building Figure 6: Stream correlation heatmaps...")

def stream_corr_matrix(dataset_prefix, predictions_dir='results/predictions'):
    """Return 3x3 Pearson correlation matrix of NSS / SigLIP / CLIP-H preds."""
    files = {
        'NSS':    f'{predictions_dir}/{dataset_prefix}_NSS_only.csv',
        'SigLIP': f'{predictions_dir}/{dataset_prefix}_SigLIP_only.csv',
        'CLIP-H': f'{predictions_dir}/{dataset_prefix}_CLIP-H_only.csv',
    }
    preds = {}
    for k, f in files.items():
        if not os.path.exists(f):
            print(f"    Missing: {f}")
            return None, None
        df = pd.read_csv(f)
        # Align by image name
        df = df.sort_values('image_name').reset_index(drop=True)
        preds[k] = df['pred_mos'].values
        ref_names = df['image_name'].values

    streams = ['NSS', 'SigLIP', 'CLIP-H']
    n = len(streams)
    corr = np.zeros((n, n))
    for i, a in enumerate(streams):
        for j, b in enumerate(streams):
            corr[i, j] = stats.pearsonr(preds[a], preds[b])[0]
    return corr, streams

datasets_to_plot = [
    ('koniq',          'KonIQ-10k'),
    ('kadid',          'KADID-10k'),
    ('liveitw_koniq',  'LIVE-itW (cross-dataset)'),
]

fig, axes = plt.subplots(1, len(datasets_to_plot),
                         figsize=(5 * len(datasets_to_plot), 4.5))

for ax, (prefix, title) in zip(axes, datasets_to_plot):
    corr, streams = stream_corr_matrix(prefix)
    if corr is None:
        # Fallback for LIVE-itW: try intra or finetune variants
        if prefix == 'liveitw_koniq':
            for alt in ['liveitw_intra', 'liveitw_finetune', 'liveitw']:
                corr, streams = stream_corr_matrix(alt)
                if corr is not None:
                    title = f'LIVE-itW ({alt.replace("liveitw_", "")})'
                    break

    if corr is None:
        ax.text(0.5, 0.5, f'{title}\nfiles not found',
                transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')
        continue

    im = ax.imshow(corr, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='equal')

    # Annotate
    for i in range(len(streams)):
        for j in range(len(streams)):
            text_color = 'white' if corr[i, j] < 0.7 else 'black'
            ax.text(j, i, f'{corr[i, j]:.3f}',
                    ha='center', va='center', fontsize=11,
                    color=text_color, fontweight='bold')

    ax.set_xticks(range(len(streams)))
    ax.set_yticks(range(len(streams)))
    ax.set_xticklabels(streams)
    ax.set_yticklabels(streams)
    ax.set_title(title, fontsize=12, pad=10)

    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Pearson correlation')

plt.suptitle('Pearson correlation between single-stream predictions',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig6_stream_correlation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig6_stream_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig6_stream_correlation.{pdf,png}")


# =============================================================
# FIGURE 9: Distortion type x severity heatmap (KADID)
# =============================================================
print("\nBuilding Figure 9: Distortion x severity heatmap...")

preds_full = pd.read_csv('results/predictions/kadid_NSS_and_SigLIP_and_CLIP-H.csv')

# Compute SROCC for each (type, level) combination
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

# Compute per-type SROCC instead of per-(type, level) since each (type, level)
# has only 81 samples (one per reference image), enough for SROCC.
n_types  = 25
n_levels = 5
heatmap  = np.full((n_types, n_levels), np.nan)

for dist_t in range(1, n_types + 1):
    for level in range(1, n_levels + 1):
        sub = preds_full[(preds_full['distortion_type']  == dist_t) &
                         (preds_full['distortion_level'] == level)]
        if len(sub) >= 10:
            sr = stats.spearmanr(sub['true_mos'], sub['pred_mos']).statistic
            heatmap[dist_t - 1, level - 1] = sr

# Sort rows by mean SROCC across severity levels (best at top)
row_means = np.nanmean(heatmap, axis=1)
sort_idx = np.argsort(row_means)[::-1]  # descending
heatmap_sorted = heatmap[sort_idx]
labels_sorted = [f"{i+1:>2}: {DISTORTION_NAMES[i+1]}" for i in sort_idx]

fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(heatmap_sorted, cmap='RdYlGn', vmin=0.0, vmax=1.0, aspect='auto')

# Annotate cells
for i in range(n_types):
    for j in range(n_levels):
        v = heatmap_sorted[i, j]
        if np.isnan(v):
            continue
        color = 'white' if v < 0.5 else 'black'
        ax.text(j, i, f'{v:.2f}',
                ha='center', va='center', fontsize=7,
                color=color)

ax.set_xticks(range(n_levels))
ax.set_xticklabels([f'Level {l+1}' for l in range(n_levels)])
ax.set_yticks(range(n_types))
ax.set_yticklabels(labels_sorted, fontsize=8)
ax.set_xlabel('Distortion severity (1 = mildest, 5 = severest)', fontsize=11)
ax.set_title('SROCC per (distortion type, severity level) on KADID-10k\nFull NSS + SigLIP + CLIP-H model', fontsize=11, pad=10)

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label('SROCC', fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig9_distortion_severity_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig9_distortion_severity_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig9_distortion_severity_heatmap.{pdf,png}")


# =============================================================
# FIGURE 10: Qualitative examples grid
# =============================================================
# 2 rows x 4 cols: row 1 = best predictions, row 2 = worst predictions
# Pick examples from KADID (we have distortion labels, easier to caption)

print("\nBuilding Figure 10: Qualitative examples...")

# Helper to find image path (look in standard locations)
def find_image_path(image_name, dataset='kadid'):
    candidates = []
    if dataset == 'kadid':
        candidates = [
            f'data/kadid10k/images/{image_name}',
            f'D:/Multimodal LLMs for IQA/data/kadid10k/images/{image_name}',
            f'../data/kadid10k/images/{image_name}',
        ]
    elif dataset == 'koniq':
        candidates = [
            f'data/koniq10k/images/{image_name}',
            f'D:/Multimodal LLMs for IQA/data/koniq10k/images/{image_name}',
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# Use KADID predictions (richest labels)
preds = pd.read_csv('results/predictions/kadid_NSS_and_SigLIP_and_CLIP-H.csv')
preds['abs_error'] = (preds['true_mos'] - preds['pred_mos']).abs()

# Best predictions: 4 lowest abs_error spanning quality range
# Sort by abs_error ascending, then pick a diverse 4
best_candidates = preds.sort_values('abs_error').head(200)
# Pick across MOS bins
mos_bins = np.linspace(best_candidates['true_mos'].min(),
                       best_candidates['true_mos'].max(), 5)
best_picks = []
for i in range(4):
    lo, hi = mos_bins[i], mos_bins[i + 1]
    in_bin = best_candidates[(best_candidates['true_mos'] >= lo) &
                              (best_candidates['true_mos'] < hi)]
    if len(in_bin) > 0:
        best_picks.append(in_bin.iloc[0])

# Worst predictions: 4 highest abs_error
worst_picks = preds.sort_values('abs_error', ascending=False).head(4).iterrows()
worst_picks = [r for _, r in worst_picks]

print(f"  Selected {len(best_picks)} best, {len(worst_picks)} worst examples")

# Build grid
fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))

DIST_NAMES_SHORT = {
    1: "Gauss blur",  2: "Lens blur",  3: "Motion blur",  4: "Color diff",
    5: "Color shift", 6: "Color quant", 7: "Color sat 1", 8: "Color sat 2",
    9: "JPEG2000",    10: "JPEG",       11: "White noise", 12: "WN CC",
    13: "Impulse",    14: "Mult noise", 15: "Denoise",   16: "Brighten",
    17: "Darken",     18: "Mean shift", 19: "Jitter",     20: "Non-ecc",
    21: "Pixelate",   22: "Quant",      23: "Color block", 24: "Sharpen",
    25: "Contrast",
}

def render_panel(ax, row, label_prefix, missing_count):
    img_path = find_image_path(row['image_name'], dataset='kadid')
    if img_path is None:
        ax.text(0.5, 0.5,
                f'{label_prefix}\n{row["image_name"]}\n(image not found)\n\nTrue: {row["true_mos"]:.2f}\nPred: {row["pred_mos"]:.2f}',
                ha='center', va='center', fontsize=10,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return missing_count + 1

    img = Image.open(img_path).convert('RGB')
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    dist_t = int(row.get('distortion_type', -1))
    dist_l = int(row.get('distortion_level', -1))
    title = f"{label_prefix}\nTrue: {row['true_mos']:.2f}  Pred: {row['pred_mos']:.2f}"
    if dist_t in DIST_NAMES_SHORT and dist_l > 0:
        title += f"\n{DIST_NAMES_SHORT[dist_t]} (lvl {dist_l})"
    title += f"  |error|: {row['abs_error']:.2f}"
    ax.set_title(title, fontsize=10)
    return missing_count

missing = 0
for i, row in enumerate(best_picks):
    missing = render_panel(axes[0, i], row, f"BEST {i+1}", missing)
for i, row in enumerate(worst_picks):
    missing = render_panel(axes[1, i], row, f"WORST {i+1}", missing)

# If row not full
for i in range(len(best_picks), 4):
    axes[0, i].axis('off')
for i in range(len(worst_picks), 4):
    axes[1, i].axis('off')

if missing > 0:
    plt.suptitle(f"Qualitative examples on KADID-10k. NOTE: {missing} image(s) could not be loaded.\n"
                 f"Update find_image_path() with your image folder.",
                 fontsize=12, y=1.02, color='red')
else:
    plt.suptitle("Qualitative examples on KADID-10k: best and worst predictions\n"
                 "(NSS + SigLIP + CLIP-H model)",
                 fontsize=13, y=1.02)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig10_qualitative_examples.pdf', dpi=200, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig10_qualitative_examples.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> saved fig10_qualitative_examples.{{pdf,png}} ({missing} image(s) missing)")

if missing > 0:
    print(f"\n  WARNING: {missing} image(s) could not be loaded.")
    print(f"  Edit find_image_path() in this script to point to your KADID images folder.")
    print(f"  Currently checked: data/kadid10k/images/, D:/Multimodal LLMs for IQA/data/kadid10k/images/")


# =============================================================
print("\n" + "=" * 60)
print("All figures generated in results/figures/")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    size_kb = os.path.getsize(f'{FIG_DIR}/{f}') / 1024
    print(f"  {f}: {size_kb:.0f} KB")
