"""
make_paper_figures.py - Generate Figures 2, 3, 4, 5, 7 for the paper
Reads from results/ folder and saves to results/figures/

Fig. 2: Per-distortion SROCC (horizontal bars, KADID)
Fig. 3: NSS contribution per distortion (horizontal bars, KADID)
Fig. 4: Predicted vs actual scatter (3 subplots: KonIQ, KADID, LIVE-itW)
Fig. 5: Ablation curves (line plot, 3 datasets)
Fig. 7: Per-severity SROCC on KADID (line plot, severity 1-5)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy import stats

# ===== Style settings =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
})

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================
# FIGURE 2: KADID per-distortion SROCC bar chart
# =============================================================
print("Building Figure 2: Per-distortion SROCC...")

per_dist = pd.read_csv('results/kadid_per_distortion_results.csv')
per_dist = per_dist.sort_values('srocc', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 7))

def get_color(s):
    if s < 0.85:  return '#d73027'
    elif s < 0.95: return '#fdae61'
    else:         return '#1a9850'

colors = [get_color(s) for s in per_dist['srocc']]

y_pos = np.arange(len(per_dist))
bars = ax.barh(y_pos, per_dist['srocc'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['distortion_type']:>2}: {r['distortion_name']}"
                    for _, r in per_dist.iterrows()])

ax.set_xlim(0.4, 1.0)
ax.set_xlabel('SROCC', fontsize=11)
ax.set_title('Per-distortion SROCC on KADID-10k (NSS + SigLIP + CLIP-H)',
             fontsize=12, pad=10)
ax.axvline(x=0.85, color='gray', linestyle=':', linewidth=0.7, alpha=0.7)
ax.axvline(x=0.95, color='gray', linestyle=':', linewidth=0.7, alpha=0.7)
ax.grid(True, axis='x')
ax.set_axisbelow(True)

for i, s in enumerate(per_dist['srocc']):
    ax.text(s + 0.005, i, f'{s:.3f}', va='center', fontsize=8)

legend_elems = [
    Patch(facecolor='#1a9850', edgecolor='black', label='SROCC ≥ 0.95'),
    Patch(facecolor='#fdae61', edgecolor='black', label='0.85 ≤ SROCC < 0.95'),
    Patch(facecolor='#d73027', edgecolor='black', label='SROCC < 0.85'),
]
ax.legend(handles=legend_elems, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2_per_distortion_srocc.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig2_per_distortion_srocc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig2_per_distortion_srocc.{pdf,png}")


# =============================================================
# FIGURE 3: NSS contribution per distortion (delta SROCC)
# =============================================================
print("Building Figure 3: NSS contribution per distortion...")

nss_contrib = pd.read_csv('results/kadid_nss_contribution_per_distortion.csv')
nss_contrib = nss_contrib.sort_values('delta_srocc', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 7))

colors = ['#1a9850' if d > 0 else '#d73027' for d in nss_contrib['delta_srocc']]

y_pos = np.arange(len(nss_contrib))
bars = ax.barh(y_pos, nss_contrib['delta_srocc'], color=colors,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['distortion_type']:>2}: {r['distortion_name']}"
                    for _, r in nss_contrib.iterrows()])

ax.axvline(x=0, color='black', linewidth=1)
ax.set_xlabel('Δ SROCC (with NSS - without NSS)', fontsize=11)
ax.set_title('NSS contribution per distortion type on KADID-10k',
             fontsize=12, pad=10)
ax.grid(True, axis='x')
ax.set_axisbelow(True)

for i, d in enumerate(nss_contrib['delta_srocc']):
    if d > 0:
        ax.text(d + 0.0005, i, f'+{d:.4f}', va='center', fontsize=8)
    else:
        ax.text(d - 0.0005, i, f'{d:.4f}', va='center', ha='right', fontsize=8)

legend_elems = [
    Patch(facecolor='#1a9850', edgecolor='black', label='NSS helps'),
    Patch(facecolor='#d73027', edgecolor='black', label='NSS hurts'),
]
ax.legend(handles=legend_elems, loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_nss_contribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig3_nss_contribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig3_nss_contribution.{pdf,png}")


# =============================================================
# FIGURE 4: Predicted vs actual scatter plots (3 subplots)
# =============================================================
print("Building Figure 4: Predicted vs actual scatter plots...")

koniq_csv = 'results/predictions/koniq_NSS_and_SigLIP_and_CLIP-H.csv'
kadid_csv = 'results/predictions/kadid_NSS_and_SigLIP_and_CLIP-H.csv'
live_csv  = 'results/predictions/liveitw_finetune_NSS_and_SigLIP_and_CLIP-H.csv'

if not os.path.exists(koniq_csv):
    candidates = [f for f in os.listdir('results/predictions')
                  if f.startswith('koniq') and 'NSS' in f and 'CLIP' in f]
    if candidates: koniq_csv = f'results/predictions/{candidates[0]}'

if not os.path.exists(live_csv):
    candidates = [f for f in os.listdir('results/predictions')
                  if f.startswith('liveitw') and 'NSS' in f and 'CLIP' in f]
    if candidates: live_csv = f'results/predictions/{candidates[0]}'

print(f"  KonIQ:    {koniq_csv}")
print(f"  KADID:    {kadid_csv}")
print(f"  LIVE-itW: {live_csv}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [
    ('KonIQ-10k',  koniq_csv, axes[0]),
    ('KADID-10k',  kadid_csv, axes[1]),
    ('LIVE-itW',   live_csv,  axes[2]),
]

for name, csv_path, ax in datasets:
    if not os.path.exists(csv_path):
        ax.text(0.5, 0.5, f'{name}\nfile not found',
                transform=ax.transAxes, ha='center', va='center')
        continue

    df = pd.read_csv(csv_path)
    y_true = df['true_mos'].values
    y_pred = df['pred_mos'].values

    sr = stats.spearmanr(y_true, y_pred).statistic
    pl = stats.pearsonr(y_true, y_pred)[0]

    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color='steelblue',
               edgecolors='none', rasterized=True)

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            'k--', linewidth=1.0, alpha=0.6, label='y = x')

    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, p(xs), 'r-', linewidth=1.2, alpha=0.8, label='Linear fit')

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Ground-truth MOS', fontsize=11)
    ax.set_ylabel('Predicted MOS', fontsize=11)
    ax.set_title(f'{name}\nSROCC = {sr:.4f}, PLCC = {pl:.4f}', fontsize=11)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.set_aspect('equal', 'box')

plt.suptitle('Predicted vs Ground-Truth MOS for the Three-Stream Fusion Model',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig4_pred_vs_actual.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig4_pred_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig4_pred_vs_actual.{pdf,png}")


# =============================================================
# FIGURE 5: Ablation as CURVES across 3 datasets
# =============================================================
print("Building Figure 5: Ablation curves...")

koniq_results = pd.read_csv('results/koniq_fusion_mlp_v3_results.csv')
kadid_results = pd.read_csv('results/kadid_fusion_mlp_v3_results.csv')

# Use KonIQ-only LIVE-itW results for fair comparison with KonIQ/KADID intra-dataset
live_results = None
live_label = ''
for candidate, label in [
    ('results/liveitw_koniq_only_results.csv',     'LIVE-itW (cross-dataset)'),
    ('results/liveitw_finetune_results.csv',       'LIVE-itW (pretrain+finetune)'),
    ('results/liveitw_fusion_mlp_v3_results.csv',  'LIVE-itW'),
]:
    if os.path.exists(candidate):
        live_results = pd.read_csv(candidate)
        live_label = label
        print(f"  Using LIVE-itW results from: {candidate}")
        break

# Canonical order: complexity-progression
canonical_order = [
    'NSS only',
    'SigLIP only',
    'CLIP-H only',
    'NSS + SigLIP',
    'NSS + CLIP-H',
    'SigLIP + CLIP-H',
    'NSS + SigLIP + CLIP-H',
]
short_labels = [
    'NSS',
    'SigLIP',
    'CLIP-H',
    'NSS\n+ SigLIP',
    'NSS\n+ CLIP-H',
    'SigLIP\n+ CLIP-H',
    'All three',
]

def reorder(df):
    return df.set_index('model').reindex(canonical_order).reset_index()

koniq_results = reorder(koniq_results)
kadid_results = reorder(kadid_results)
if live_results is not None:
    live_results = reorder(live_results)

def get_srocc(df):
    if 'srocc_mean' in df.columns:
        return df['srocc_mean'].values, (df['srocc_std'].values
                                          if 'srocc_std' in df.columns else None)
    elif 'srocc' in df.columns:
        return df['srocc'].values, None
    return None, None

koniq_srocc, koniq_std = get_srocc(koniq_results)
kadid_srocc, kadid_std = get_srocc(kadid_results)
live_srocc,  live_std  = (get_srocc(live_results) if live_results is not None
                           else (None, None))

x = np.arange(len(canonical_order))

fig, ax = plt.subplots(figsize=(10, 5.5))

# KADID first (highest line)
ax.plot(x, kadid_srocc, marker='o', markersize=8, linewidth=2,
        color='#1a9850', label='KADID-10k')
if kadid_std is not None:
    ax.fill_between(x, kadid_srocc - kadid_std, kadid_srocc + kadid_std,
                    color='#1a9850', alpha=0.15)

# KonIQ
ax.plot(x, koniq_srocc, marker='s', markersize=8, linewidth=2,
        color='#4575b4', label='KonIQ-10k')
if koniq_std is not None:
    ax.fill_between(x, koniq_srocc - koniq_std, koniq_srocc + koniq_std,
                    color='#4575b4', alpha=0.15)

# LIVE-itW
if live_srocc is not None:
    ax.plot(x, live_srocc, marker='^', markersize=8, linewidth=2,
            color='#fc8d59', label=live_label)
    if live_std is not None:
        ax.fill_between(x, live_srocc - live_std, live_srocc + live_std,
                        color='#fc8d59', alpha=0.15)

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylabel('SROCC', fontsize=11)
ax.set_title('Stream Ablation: SROCC across Three Datasets', fontsize=12, pad=10)
ax.set_ylim(0.4, 1.0)
ax.grid(True)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=True)

# Annotate end points (rightmost variant) with values
ax.annotate(f'{kadid_srocc[-1]:.4f}',
            xy=(x[-1], kadid_srocc[-1]),
            xytext=(8, 4), textcoords='offset points',
            fontsize=9, color='#1a9850', fontweight='bold')
ax.annotate(f'{koniq_srocc[-1]:.4f}',
            xy=(x[-1], koniq_srocc[-1]),
            xytext=(8, 4), textcoords='offset points',
            fontsize=9, color='#4575b4', fontweight='bold')
if live_srocc is not None:
    ax.annotate(f'{live_srocc[-1]:.4f}',
                xy=(x[-1], live_srocc[-1]),
                xytext=(8, -12), textcoords='offset points',
                fontsize=9, color='#fc8d59', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig5_ablation_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig5_ablation_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig5_ablation_curves.{pdf,png}")


# =============================================================
# FIGURE 7: Per-severity-level SROCC on KADID
# =============================================================
print("Building Figure 7: Per-severity-level SROCC curves...")

# Load full predictions from KADID
preds_full = pd.read_csv('results/predictions/kadid_NSS_and_SigLIP_and_CLIP-H.csv')

# Compute per-severity-level SROCC: pool across all distortion types,
# then split by severity level.
def srocc_by_severity(df):
    results = {}
    for level in sorted(df['distortion_level'].unique()):
        if level < 1 or level > 5:
            continue
        sub = df[df['distortion_level'] == level]
        if len(sub) < 10:
            continue
        sr = stats.spearmanr(sub['true_mos'], sub['pred_mos']).statistic
        pl = stats.pearsonr(sub['true_mos'], sub['pred_mos'])[0]
        results[int(level)] = (sr, pl, len(sub))
    return results

# We need predictions for each ablation variant to compare.
# Load whichever variants exist:
variants_to_compare = [
    ('kadid_NSS_only.csv',                            'NSS only',              '#999999'),
    ('kadid_SigLIP_only.csv',                         'SigLIP only',           '#4575b4'),
    ('kadid_CLIP-H_only.csv',                         'CLIP-H only',           '#fc8d59'),
    ('kadid_SigLIP_and_CLIP-H.csv',                   'SigLIP + CLIP-H',       '#fdae61'),
    ('kadid_NSS_and_SigLIP_and_CLIP-H.csv',           'NSS + SigLIP + CLIP-H', '#1a9850'),
]

fig, ax = plt.subplots(figsize=(8, 5.5))

for fname, label, color in variants_to_compare:
    fpath = f'results/predictions/{fname}'
    if not os.path.exists(fpath):
        print(f"  Skipping {fname} (not found)")
        continue
    df = pd.read_csv(fpath)
    by_sev = srocc_by_severity(df)
    levels = sorted(by_sev.keys())
    sroccs = [by_sev[l][0] for l in levels]

    is_full = (label == 'NSS + SigLIP + CLIP-H')
    ax.plot(levels, sroccs,
            marker='o', markersize=7,
            linewidth=2.5 if is_full else 1.3,
            color=color, label=label,
            alpha=1.0 if is_full else 0.75)

ax.set_xlabel('Distortion severity level (KADID)', fontsize=11)
ax.set_ylabel('SROCC', fontsize=11)
ax.set_title('SROCC vs distortion severity level on KADID-10k',
             fontsize=12, pad=10)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['1\n(mildest)', '2', '3', '4', '5\n(severest)'])
ax.set_ylim(0.4, 1.0)
ax.grid(True)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig7_per_severity_srocc.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig7_per_severity_srocc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> saved fig7_per_severity_srocc.{pdf,png}")

# =============================================================
print("\n" + "=" * 60)
print("All figures generated in results/figures/")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    size_kb = os.path.getsize(f'{FIG_DIR}/{f}') / 1024
    print(f"  {f}: {size_kb:.0f} KB")
