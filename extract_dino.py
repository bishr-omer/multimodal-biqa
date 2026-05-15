import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model = model.to(device).eval()

# Standard DINOv2 preprocessing (ImageNet stats, 224px)
_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225],
)

full_preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _normalize,
])

crop_preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    _normalize,
])

def multi_crop_features(pil_img):
    """Full image + FiveCrop → mean-pool → L2-normalize → 1024-dim vector."""
    w, h = pil_img.size
    crop_size = int(min(h, w) * 0.85)
    crop_size = max(crop_size, 64)

    five_crop = transforms.FiveCrop(crop_size)
    crops = list(five_crop(pil_img))  # 5 PIL images

    full_t  = full_preprocess(pil_img).unsqueeze(0)                        # (1, 3, 224, 224)
    crops_t = torch.stack([crop_preprocess(c) for c in crops])              # (5, 3, 224, 224)
    tensors = torch.cat([full_t, crops_t], dim=0).to(device)               # (6, 3, 224, 224)

    with torch.no_grad():
        feats = model(tensors)                                               # (6, 1024) — CLS token
        feat_avg = feats.mean(dim=0, keepdim=True)                          # (1, 1024)
        feat_avg = feat_avg / feat_avg.norm(dim=-1, keepdim=True)
    return feat_avg.squeeze().cpu().numpy()

def extract_features(image_folder, output_path):
    image_folder = Path(image_folder)
    image_paths = sorted(
        list(image_folder.glob("*.jpg")) +
        list(image_folder.glob("*.JPG")) +
        list(image_folder.glob("*.png")) +
        list(image_folder.glob("*.bmp")) +
        list(image_folder.glob("*.BMP"))
    )
    print(f"Found {len(image_paths)} images in {image_folder}")
    print(f"Device: {device}")

    names, features = [], []
    for img_path in tqdm(image_paths, desc="Extracting DINOv2 features"):
        try:
            img = Image.open(img_path).convert("RGB")
            feat = multi_crop_features(img)
            names.append(img_path.name)
            features.append(feat)
        except Exception as e:
            print(f"Skipped {img_path.name}: {e}")

    features = np.array(features)
    np.save(output_path, {"names": names, "features": features})
    print(f"Saved {len(names)} vectors → shape {features.shape}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        extract_features(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python extract_dino.py <image_folder> <output_path>")
