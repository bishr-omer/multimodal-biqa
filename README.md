<div align="center">

# Distortion-Aware Fusion of Statistical and Vision-Language Features<br>for Blind Image Quality Assessment

**Bishr Omer Abdelrahman · Xu Li**

Northwestern Polytechnical University, Xi'an, China

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/bishr-omer/multimodal-biqa?style=social)](https://github.com/bishr-omer/multimodal-biqa)

</div>

---

## Overview

We propose a **three-stream BIQA framework** that fuses classical natural scene statistics with two complementary vision-language models through a lightweight multiplicative gating mechanism.

| Stream | Backbone | Output |
|--------|----------|--------|
| **NSS** | Spatial + spectral statistics | 138D |
| **SigLIP** | ViT-SO400M-14-SigLIP-384 | 1152D → 256D (PCA) |
| **CLIP-H** | ViT-H-14 LAION-2B | 1024D → 256D (PCA) |

- All VLM backbones are **entirely frozen** — no end-to-end fine-tuning required
- Only the MLP regression head (**~466,000 parameters**) is trained
- Full pipeline runs on **CPU** after a one-time GPU feature extraction step
- A multiplicative gating network learns per-input stream weights; gate values correlate positively (**ρ = 0.33**) with per-distortion NSS contribution

---

## Results

<div align="center">

### State-of-the-Art Comparison

| Method | KonIQ SROCC | KADID SROCC | LIVE-itW SROCC |
|--------|-------------|-------------|----------------|
| BRISQUE | 0.665 | 0.528 | 0.561 |
| HyperIQA | 0.906 | 0.852 | 0.852 |
| MUSIQ | 0.916 | 0.872 | 0.847 |
| MANIQA | 0.923 | 0.946 | 0.853 |
| LIQE | 0.919 | 0.930 | 0.870 |
| Q-Align† | 0.940 | 0.937 | 0.886 |
| **Ours** | **0.9142** | **0.9715** | **0.8527** |

† end-to-end fine-tuned backbone

</div>

> Our method achieves **state-of-the-art SROCC of 0.9715 on KADID-10k**, surpassing all comparison methods including Q-Align, while keeping VLM backbones frozen.

---

## Figures

<table>
<tr>
<td width="50%">

**Per-distortion SROCC on KADID-10k**
![Per-distortion SROCC](results/figures/fig2_per_distortion_srocc.png)

</td>
<td width="50%">

**NSS contribution per distortion type**
![NSS contribution](results/figures/fig3_nss_contribution.png)

</td>
</tr>
</table>

**Distortion-aware gate analysis**
![Gate analysis](results/figures/fig_gate_analysis.png)

---

## Installation

```bash
git clone https://github.com/bishr-omer/multimodal-biqa.git
cd multimodal-biqa
pip install -r requirements.txt
```

> Tested on Python 3.11. GPU required only for feature extraction. All training runs on CPU.

---

## Quick Start

### Step 1 — Extract NSS features (CPU)

```bash
python extract_nss.py \
    --dataset koniq \
    --image_dir /path/to/koniq/images \
    --output_path features/koniq_nss.npy
```

### Step 2 — Extract VLM features (GPU)

```bash
python extract_clip_h14.py --dataset koniq --image_dir /path/to/koniq/images
```

> **Skip this step** if using precomputed features (see below).

### Step 3 — Train static fusion

```bash
python train_fusion.py --config configs/koniq.yaml
python train_fusion.py --config configs/kadid.yaml
python train_fusion_liveitw.py --config configs/liveitw.yaml
```

### Step 4 — Train gating model

```bash
python train_gated.py --config configs/koniq.yaml
python train_gated.py --config configs/kadid.yaml
```

---

## Precomputed Features

Skip GPU feature extraction by downloading precomputed `.npy` files for all three datasets:

<div align="center">

**[⬇ Download Precomputed Features (~250 MB, Google Drive)](https://drive.google.com/drive/folders/1Bk3ABv7LkEqYNyAk1lA7h2ffQGpX0uH9?usp=drive_link)**

</div>

Place downloaded files in the `features/` directory.

---

## Repository Structure

    multimodal-biqa/
    ├── extract_nss.py               # NSS feature extraction (CPU)
    ├── extract_mvg.py               # Alternative NSS extractor
    ├── extract_clip_h14.py          # CLIP-H feature extraction (GPU)
    ├── extract_dino.py              # DINOv2 feature extraction (GPU)
    ├── train_fusion.py              # Static fusion, 5-fold CV
    ├── fusion_kadid_mlp_v3.py       # KADID fusion with per-distortion analysis
    ├── train_fusion_liveitw.py      # Pretrain + finetune + ensemble for LIVE-itW
    ├── fusion_liveitw_koniq_only.py # KonIQ to LIVE-itW cross-dataset transfer
    ├── train_gated.py               # Multiplicative gating fusion
    ├── fig_gate_analysis.py         # Gate analysis figure
    ├── make_paper_figures.py        # Paper figures
    ├── loss_ablation_experiment.py  # Loss ablation on KonIQ
    ├── loss_ablation_liveitw.py     # Loss ablation on LIVE-itW
    ├── requirements.txt
    ├── configs/
    │   ├── koniq.yaml
    │   ├── kadid.yaml
    │   └── liveitw.yaml
    ├── figures/
    └── results/

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{adam2026distortion,
  title   = {Distortion-Aware Fusion of Statistical and Vision-Language
             Features for Blind Image Quality Assessment},
  author  = {Adam, Bishr Omer and Li, Xu},
  year    = {2026}
}
```

---

## Acknowledgements

VLM feature extraction uses [OpenCLIP](https://github.com/mlfoundations/open_clip)
and [Hugging Face Transformers](https://github.com/huggingface/transformers).
