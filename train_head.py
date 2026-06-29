"""
Train the MLP regression head on cached frozen features.

Uses a single train/test split (image-level by default). For the full SOTA
comparison we use 5-fold cross-validation on KonIQ; pass --folds 5 for that.

Usage:
    python train_head.py --features features/koniq.npz --epochs 80
"""
import argparse, numpy as np, torch
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from biqa_common import make_head, hybrid_loss, image_level_split


def train_eval(X, y, tr, te, epochs=80, lr=5e-4, device="cpu"):
    sc = StandardScaler().fit(X[tr])
    Xtr = torch.tensor(sc.transform(X[tr]), dtype=torch.float32, device=device)
    Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], dtype=torch.float32, device=device)
    head = make_head(X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            hybrid_loss(head(Xtr[b]).squeeze(-1), ytr[b]).backward()
            opt.step()
        sched.step()
    head.eval()
    with torch.no_grad():
        pr = head(Xte).squeeze(-1).cpu().numpy()
    return spearmanr(pr, y[te]).correlation, pearsonr(pr, y[te])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=1, help="set 5 for 5-fold CV")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.features, allow_pickle=True)
    X, y = d["X"], d["y"]

    srccs, plccs = [], []
    for f in range(args.folds):
        tr, te = image_level_split(len(X), seed=args.seed + f)
        s, p = train_eval(X, y, tr, te, epochs=args.epochs, device=device)
        srccs.append(s); plccs.append(p)
        print(f"fold {f}: SROCC {s:.4f}  PLCC {p:.4f}")
    print(f"\nmean SROCC {np.mean(srccs):.4f}  PLCC {np.mean(plccs):.4f}")


if __name__ == "__main__":
    main()
