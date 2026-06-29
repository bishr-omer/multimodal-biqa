"""
End-to-end reproducibility test on a SINGLE dataset.

Runs the whole pipeline in one process to confirm everything works locally
before committing: load -> extract frozen SigLIP features -> train frozen head
-> evaluate under image-level and reference-level splits -> LoRA adaptation.

This is a SMOKE TEST, not the repo's main interface (use the modular scripts
extract_features.py / train_head.py / evaluate_protocols.py / lora_adapt.py for
real runs). CSIQ is recommended here: small (~866 images), fast, has reference
structure.

Usage:
    python test_pipeline.py --dataset csiq \
        --meta_dir meta_info --img_root /path/to/csiq_images
"""
import argparse, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from peft import LoraConfig, get_peft_model

from biqa_common import (load_dataset, embed_image, make_head, hybrid_loss,
                         image_level_split, reference_level_split, SIGLIP_CKPT)


def train_head_eval(X, y, tr, te, epochs, device):
    sc = StandardScaler().fit(X[tr])
    Xtr = torch.tensor(sc.transform(X[tr]), dtype=torch.float32, device=device)
    Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], dtype=torch.float32, device=device)
    head = make_head(X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-4)
    for _ in range(epochs):
        head.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 128):
            b = perm[i:i+128]; opt.zero_grad()
            hybrid_loss(head(Xtr[b]).squeeze(-1), ytr[b]).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return spearmanr(head(Xte).squeeze(-1).cpu().numpy(), y[te]).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="csiq")
    ap.add_argument("--meta_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--head_epochs", type=int, default=80)
    ap.add_argument("--lora_epochs", type=int, default=5)
    ap.add_argument("--save_lora", default=None, help="dir to save LoRA adapter weights")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_CKPT)

    # ---- 1. load ----
    paths, refs, y = load_dataset(args.dataset, args.meta_dir, args.img_root)
    print(f"[load] {len(paths)} images, {len(set(refs))} references")
    assert len(paths) > 0, "no images matched -- check img_root and meta_dir"

    # ---- 2. extract frozen features ----
    print("[extract] frozen SigLIP five-crop features ...")
    model = SiglipVisionModel.from_pretrained(SIGLIP_CKPT).to(device).eval()
    X = np.zeros((len(paths), 1152), dtype="float32")
    t0 = time.time()
    for i, p in enumerate(paths):
        X[i] = embed_image(p, model, processor, device)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(paths)}  {time.time()-t0:.0f}s")
    print(f"[extract] done in {time.time()-t0:.0f}s")

    # ---- 3. frozen head, both protocols ----
    tr_i, te_i = image_level_split(len(X))
    img_s = train_head_eval(X, y, tr_i, te_i, args.head_epochs, device)
    tr_r, te_r = reference_level_split(refs)
    ref_s = train_head_eval(X, y, tr_r, te_r, args.head_epochs, device)
    print(f"[frozen] image-level SROCC     {img_s:.4f}")
    print(f"[frozen] reference-level SROCC {ref_s:.4f}")
    print(f"[frozen] inflation gap         {img_s - ref_s:.4f}")

    # ---- 4. LoRA adaptation (reference-level) ----
    print("[lora] adapting SigLIP backbone (reference-level) ...")
    yn = (y - y.mean()) / (y.std() + 1e-8)

    class DS(Dataset):
        def __init__(self, idx): self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, k):
            j = self.idx[k]
            px = processor(images=Image.open(paths[j]).convert("RGB"),
                           return_tensors="pt")["pixel_values"][0]
            return px, yn[j]

    tr_dl = DataLoader(DS(tr_r), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    te_dl = DataLoader(DS(te_r), batch_size=16, shuffle=False, num_workers=2)

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
    for ep in range(args.lora_epochs):
        sig.train(); head.train()
        for px, yb in tr_dl:
            px, yb = px.to(device), yb.to(device); opt.zero_grad()
            with torch.autocast(device, dtype=torch.float16):
                loss = hybrid_loss(head(sig(pixel_values=px).pooler_output.float()).squeeze(-1), yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in sig.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            scaler.step(opt); scaler.update()
        sig.eval(); head.eval(); preds = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            for px, _ in te_dl:
                preds.append(head(sig(pixel_values=px.to(device)).pooler_output.float()).squeeze(-1).cpu().numpy())
        sr = spearmanr(np.concatenate(preds), y[te_r]).correlation
        best = max(best, sr)
        print(f"    ep {ep}  SROCC {sr:.4f}")

    print(f"[lora] frozen ref-level {ref_s:.4f} -> LoRA {best:.4f}  (delta {best - ref_s:+.4f})")

    if args.save_lora:
        sig.save_pretrained(args.save_lora)
        print(f"[lora] adapter weights saved to {args.save_lora}")

    print("\n=== pipeline OK ===")


if __name__ == "__main__":
    main()
