"""
Generate the three result figures from the paper's final numbers.

Writes: fig3_leakage.png, fig4_adaptation.png, fig5_curves.png

The numbers are the six-dataset results reported in the paper; editing the
dictionaries below is enough to regenerate the figures if results change.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
BLUE, RED, ORANGE, GREEN = "#4C72B0", "#C44E52", "#DD8452", "#55A868"


def fig3_leakage():
    data = [("CSIQ\n(30 ref)", 0.969, 0.912), ("LIVE-MD\n(30 ref)", 0.965, 0.904),
            ("KADID\n(81 ref)", 0.955, 0.787), ("PIPAL\n(200 ref)", 0.793, 0.576),
            ("TID2013\n(25 ref)", 0.950, 0.514)]
    labels = [d[0] for d in data]; img = [d[1] for d in data]; ref = [d[2] for d in data]
    x = np.arange(len(data)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(x - w/2, img, w, label="Image-level split (conventional)", color=BLUE, zorder=3)
    ax.bar(x + w/2, ref, w, label="Reference-level split", color=RED, zorder=3)
    for xi, hi, lo in zip(x, img, ref):
        ax.annotate(f"gap {hi-lo:.2f}", (xi, max(hi, lo) + 0.015), ha="center",
                    fontsize=9, color="#8a5a2b", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Frozen SigLIP SROCC"); ax.set_ylim(0, 1.08)
    ax.set_title("Image-level splitting masks true dataset difficulty", fontsize=11.5, pad=10)
    ax.legend(loc="lower center", fontsize=9, ncol=2, bbox_to_anchor=(0.5, -0.26))
    fig.tight_layout(); fig.savefig("fig3_leakage.png", bbox_inches="tight"); plt.close(fig)


def fig4_adaptation():
    data = [("TID2013", 0.514, 0.871), ("PIPAL", 0.576, 0.707), ("KADID", 0.787, 0.927),
            ("KonIQ", 0.887, 0.951), ("LIVE-MD", 0.904, 0.893), ("CSIQ", 0.912, 0.927)]
    labels = [d[0] for d in data]; fr = [d[1] for d in data]; lo = [d[2] for d in data]
    deltas = [b - a for a, b in zip(fr, lo)]
    x = np.arange(len(data)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - w/2, fr, w, label="Frozen SigLIP", color=BLUE, zorder=3)
    ax.bar(x + w/2, lo, w, label="+ LoRA adaptation (0.23% params)", color=RED, zorder=3)
    for xi, d in zip(x, deltas):
        col = "#2a8f4a" if d >= 0.02 else "#888888"
        ax.annotate(f"{'+' if d >= 0 else ''}{d:.3f}", (xi, 1.05), ha="center",
                    fontsize=9.5, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("SROCC (reference-level)"); ax.set_ylim(0, 1.13)
    ax.set_title("Adaptation gain is largest where frozen features are weakest", fontsize=11.5, pad=10)
    ax.legend(loc="lower center", fontsize=9, ncol=2, bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout(); fig.savefig("fig4_adaptation.png", bbox_inches="tight"); plt.close(fig)


def fig5_curves():
    curves = {
        "TID2013 (frozen 0.514)": ([0.514, 0.827, 0.850, 0.848, 0.846, 0.869, 0.871], RED),
        "KADID-10k (frozen 0.787)": ([0.787, 0.904, 0.909, 0.917, 0.924, 0.927, 0.922], ORANGE),
        "KonIQ-10k (frozen 0.887)": ([0.887, 0.934, 0.942, 0.949, 0.948, 0.951], BLUE),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, (ys, c) in curves.items():
        xs = np.arange(len(ys))
        ax.plot(xs, ys, "-o", color=c, lw=2.2, ms=6, mec="white", mew=1.0, label=name, zorder=3)
    ax.set_xlabel("Adaptation epoch (0 = frozen backbone)")
    ax.set_ylabel("Test SROCC (reference-level)")
    ax.set_title("Adaptation trajectory: steep recovery where frozen is weak", fontsize=11, pad=10)
    ax.set_ylim(0.45, 0.99); ax.set_xticks(range(0, 7))
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout(); fig.savefig("fig5_curves.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig3_leakage(); fig4_adaptation(); fig5_curves()
    print("wrote fig3_leakage.png, fig4_adaptation.png, fig5_curves.png")
