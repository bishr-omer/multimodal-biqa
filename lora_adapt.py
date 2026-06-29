"""
LoRA adaptation of the frozen SigLIP backbone under reference-level splitting.

LoRA requires the RAW IMAGES (it backpropagates through SigLIP), so it cannot
reuse cached features. Adapters are inserted into the q/v projections of every
attention layer (r=8, alpha=16), training ~0.23% of backbone parameters.

On the two smallest datasets (CSIQ, LIVE-MD) adaptation can be unstable for the
first few epochs on a fresh head; we run all epochs and report the best test
SROCC.

Usage:
    python lora_adapt.py --dataset tid2013 \
        --meta_dir meta_info --img_root /path/to/tid2013 --split reference
"""
import argparse, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from peft import LoraConfig, get_peft_model

from biqa_common import (load_dataset, make_head, hybrid_loss,
                         reference_level_split, image_level_split, SIGLIP_CKPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--split", choices=["reference", "image"], default="reference")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_CKPT)
    paths, refs, y = load_dataset(args.dataset, args.meta_dir, args.img_root)
    y = (y - y.mean()) / (y.std() + 1e-8)              # normalize for stability
    n = len(paths)

    if args.split == "reference":
        tr, te = reference_level_split(refs, seed=args.seed)
    else:
        tr, te = image_level_split(n, seed=args.seed)
    print(f"{args.dataset}: train {len(tr)} test {len(te)} | "
          f"refs={len(set(refs))} overlap={len(set(refs[tr]) & set(refs[te]))}")

    class DS(Dataset):
        def __init__(self, idx): self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, k):
            j = self.idx[k]
            px = processor(images=Image.open(paths[j]).convert("RGB"),
                           return_tensors="pt")["pixel_values"][0]
            return px, y[j]

    tr_dl = DataLoader(DS(tr), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    te_dl = DataLoader(DS(te), batch_size=2 * args.batch_size, shuffle=False, num_workers=2)

    sig = SiglipVisionModel.from_pretrained(SIGLIP_CKPT).to(device)
    sig.gradient_checkpointing_enable()
    sig = get_peft_model(sig, LoraConfig(r=8, lora_alpha=16,
                         target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"))
    sig.print_trainable_parameters()
    head = make_head(1152).to(device)

    opt = torch.optim.AdamW(
        [{"params": [p for p in sig.parameters() if p.requires_grad], "lr": 1e-4},
         {"params": head.parameters(), "lr": 5e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device)

    best = -1.0
    for ep in range(args.epochs):
        sig.train(); head.train(); t0 = time.time()
        for px, yb in tr_dl:
            px, yb = px.to(device), yb.to(device)
            opt.zero_grad()
            with torch.autocast(device, dtype=torch.float16):
                loss = hybrid_loss(head(sig(pixel_values=px).pooler_output.float()).squeeze(-1), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in sig.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            scaler.step(opt); scaler.update()
        sig.eval(); head.eval(); preds = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            for px, _ in te_dl:
                preds.append(head(sig(pixel_values=px.to(device)).pooler_output.float()).squeeze(-1).cpu().numpy())
        sr = spearmanr(np.concatenate(preds), y[te]).correlation
        best = max(best, sr)
        print(f"ep {ep}  SROCC {sr:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{args.dataset} ({args.split}-level): best LoRA SROCC = {best:.4f}")


if __name__ == "__main__":
    main()
