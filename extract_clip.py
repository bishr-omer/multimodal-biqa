import open_clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
model.eval()

def extract_features(image_folder, output_path):
    image_folder = Path(image_folder)
    image_paths  = sorted(
       list(image_folder.glob("*.jpg")) + 
    list(image_folder.glob("*.JPG")) + 
    list(image_folder.glob("*.png")) + 
    list(image_folder.glob("*.bmp")) +
    list(image_folder.glob("*.BMP"))
)
    print(f"Found {len(image_paths)} images in {image_folder}")

    names, features = [], []
    for img_path in tqdm(image_paths, desc="Extracting CLIP features"):
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            names.append(img_path.name)
            features.append(feat.squeeze().numpy())
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
        print("Usage: python extract_clip.py <image_folder> <output_path>")