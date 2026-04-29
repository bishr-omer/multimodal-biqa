# Multimodal-BIQA - Blind Image Quality Assessment via Multimodal Feature Fusion

> Bridging classical natural scene statistics and vision-language model representations for no-reference image quality assessment.

---

## Overview

Classical NR-IQA methods model **statistical regularity** (what "normal" looks like in natural images). Vision-language models learn **semantic quality representations** through language-vision alignment. This project asks whether combining both gives better blind quality prediction — and demonstrates that it does, consistently across three standard benchmarks.

We fuse three complementary feature streams:

- **NSS features** — 138-dim spatial and spectral descriptors extracted via a Log-Gabor filter bank, motivated by the multivariate Gaussian (MVG) pansharpening framework
- **CLIP embeddings** — 512-dim L2-normalised visual features from ViT-B/32
- **LLaVA hidden states** — 4096-dim penultimate-layer embeddings from LLaVA-1.6 (Mistral-7B)

Fusion is performed with a GradientBoostingRegressor trained to predict MOS scores.

---

## Results

### KonIQ-10k (authentic distortions, 10,073 images)

| Model | SROCC | PLCC | KROCC |
|---|---|---|---|
| NSS only | 0.7743 | 0.8083 | 0.5785 |
| CLIP only | 0.7728 | 0.8074 | 0.5776 |
| LLaVA score only | 0.4360 | 0.5190 | 0.3325 |
| CLIP + NSS | 0.8535 | 0.8819 | 0.6659 |
| CLIP + NSS + LLaVA score | 0.8620 | 0.8887 | 0.6770 |
| **CLIP + NSS + LLaVA feat** | **0.8698** | **0.8965** | **0.6865** |

### KADID-10k (synthetic distortions, 10,125 images)

| Model | SROCC | PLCC | KROCC |
|---|---|---|---|
| NSS only | 0.8216 | 0.8185 | 0.6341 |
| CLIP only | 0.8976 | 0.8889 | 0.7220 |
| **CLIP + NSS** | **0.9118** | **0.9035** | **0.7430** |

### LIVE-itW (in-the-wild, 1,162 images)

| Model | SROCC | PLCC | KROCC |
|---|---|---|---|
| NSS only | 0.6473 | 0.6474 | 0.4697 |
| CLIP only | 0.6681 | 0.6871 | 0.4803 |
| **CLIP + NSS** | **0.7569** | **0.7743** | **0.5602** |

> **Key finding:** LLaVA hidden-state embeddings contribute independently of the scalar quality score — appending the score alone adds marginal gain, while the full 4096-dim representation drives the strongest result on KonIQ-10k. The scalar quality score encodes no information beyond what the hidden states already capture.

---

## Repository Structure

```
mvg-vlm-iqa/
├── extract_clip.py       # CLIP ViT-B/32 feature extraction
├── extract_nss.py        # Log-Gabor NSS feature extraction
├── fusion.py             # Three-stream fusion — KonIQ-10k
├── fusion_kadid.py       # Two-stream fusion — KADID-10k
├── fusion_liveitw.py     # Two-stream fusion — LIVE-itW
├── results/              # Output .npy files (not tracked)
├── data/                 # Dataset files (not tracked)
└── notebooks/
    └── demo.ipynb
```

---

## Setup

```bash
git clone https://github.com/bishr-omer/mvg-vlm-iqa.git
cd mvg-vlm-iqa
pip install -r requirements.txt
```

---

## Usage

### 1. Extract NSS features

```bash
python extract_nss.py <image_folder> results/koniq_nss_features.npy
```

### 2. Extract CLIP features

```bash
python extract_clip.py <image_folder> results/koniq_clip_features.npy
```

### 3. Run fusion

```bash
# KonIQ-10k (three-stream, requires LLaVA features)
python fusion.py

# KADID-10k
python fusion_kadid.py

# LIVE-itW
python fusion_liveitw.py
```

---

## Datasets

| Dataset | Images | Type | Source |
|---|---|---|---|
| KonIQ-10k | 10,073 | Authentic | [link](http://database.mmsp-kn.de/koniq-10k.html) |
| KADID-10k | 10,125 | Synthetic | [link](http://database.mmsp-kn.de/kadid-10k.html) |
| LIVE-itW | 1,169 | In-the-wild | [link](https://live.ece.utexas.edu/research/ChallengeDB) |

Place datasets under `data/koniq10k/`, `data/kadid10k/`, and `data/liveitw/` respectively.

---

## LLaVA Feature Extraction

LLaVA-1.6 (Mistral-7B) requires ~14GB VRAM. Extraction was performed on Kaggle T4×2 GPU sessions. The extraction notebook will be added to `notebooks/` shortly.

The saved `.npy` dictionary contains:
- `features` — shape `(N, 4096)` hidden-state embeddings
- `scores` — shape `(N,)` predicted quality scores (1–10)
- `names` — list of filenames

---

## Citation

If you use this code, please cite:

```bibtex
@misc{omer2026mvgvlmiqa,
  title   = {Bridging Statistical and Semantic Representations for Blind Image Quality Assessment via Multimodal Feature Fusion},
  author  = {Omer, Bishr},
  year    = {2026},
 url     = {https://github.com/bishr-omer/multimodal-biqa}
}
```

---

## Related Work

- [CLIP-IQA](https://github.com/IceClear/CLIP-IQA) — Wang et al., 2023
- [Q-Align](https://github.com/Q-Future/Q-Align) — Zhang et al., 2024
- [BRISQUE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) — Mittal et al., 2012

---

## License

MIT
