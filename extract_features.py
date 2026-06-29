"""
Extract frozen SigLIP five-crop embeddings for a dataset, with checkpointing.

Features are cached to a .npz file (X, y, refs). If interrupted, the run
resumes from the last checkpoint.

Usage:
    python extract_features.py --dataset koniq \
        --meta_dir meta_info --img_root /path/to/koniq --out features/koniq.npz
"""
import os, time, argparse, numpy as np, torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from biqa_common import load_dataset, embed_image, SIGLIP_CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta_dir", required=True, help="folder with meta_info_*.csv")
    ap.add_argument("--img_root", required=True, help="root folder of the images")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_CKPT)
    model = SiglipVisionModel.from_pretrained(SIGLIP_CKPT).to(device).eval()

    paths, refs, y = load_dataset(args.dataset, args.meta_dir, args.img_root)
    n, dim = len(paths), 1152
    print(f"{args.dataset}: {n} images, {len(set(refs))} references")

    ckpt = args.out + ".ckpt.npz"
    if os.path.exists(ckpt):
        d = np.load(ckpt, allow_pickle=True)
        X, done = d["X"], int(d["done"])
        print(f"resuming from {done}/{n}")
    else:
        X, done = np.zeros((n, dim), dtype="float32"), 0

    t0 = time.time()
    for i in range(done, n):
        X[i] = embed_image(paths[i], model, processor, device)
        if (i + 1) % args.save_every == 0 or i == n - 1:
            np.savez(ckpt, X=X, done=i + 1)
            print(f"  {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, X=X, y=y, refs=refs)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("saved", args.out)


if __name__ == "__main__":
    main()
