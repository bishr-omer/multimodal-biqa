import open_clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14-336', pretrained='openai'
)
model = model.to(device).eval()

# Crop-only preprocess: resize each crop to 336 then normalize (no second center-crop)
crop_preprocess = transforms.Compose([
    transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std= (0.26862954, 0.26130258, 0.27577711),
    ),
])

def multi_crop_features(pil_img):
    """Full image + FiveCrop → mean-pool → L2-normalize → 768-dim vector."""
    w, h = pil_img.size
    crop_size = int(min(h, w) * 0.85)
    crop_size = max(crop_size, 64)

    five_crop = transforms.FiveCrop(crop_size)
    crops = [pil_img] + list(five_crop(pil_img))  # 6 PIL images

    tensors = torch.stack([crop_preprocess(c) for c in crops]).to(device)  # (6, 3, 336, 336)
    with torch.no_grad():
        feats = model.encode_image(tensors)          # (6, 768)
        feat_avg = feats.mean(dim=0, keepdim=True)   # (1, 768)
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
    for img_path in tqdm(image_paths, desc="Extracting CLIP-L features"):
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
        print("Usage: python extract_clip_large.py <image_folder> <output_path>")
