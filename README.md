<div align="center">

# Low-Rank Adaptation of Frozen Vision-Language Models<br>for Blind Image Quality Assessment

**Bishr Omer Abdelrahman Adam · Xu Li**

Northwestern Polytechnical University, Xi'an, China

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

This repository contains the code for our study of **parameter-efficient adaptation of frozen
vision-language models (VLMs) for blind image quality assessment (BIQA)**, evaluated across six
datasets under both image-level and reference-level splitting protocols.

The method has two parts:

1. **An efficient frozen-feature system.** A 138-dimensional natural-scene-statistics (NSS)
   descriptor is fused with frozen **SigLIP** and **CLIP-H** embeddings through a lightweight MLP
   regression head trained on cached features. No end-to-end fine-tuning of the VLM backbones is
   required, and the head trains quickly even on CPU.

2. **Parameter-efficient backbone adaptation.** We then apply **Low-Rank Adaptation (LoRA)** to the
   SigLIP backbone, training only **0.23%** of its parameters, and study when this adaptation helps.

| Stream | Backbone | Output |
|---|---|---|
| **NSS** | Spatial + color/frequency statistics | 138D |
| **SigLIP** | ViT-SO400M-14-SigLIP-384 | 1152D → 256D (PCA) |
| **CLIP-H** | ViT-H-14 LAION-2B | 1024D → 256D (PCA) |

---

## Key Findings

- LoRA adaptation of frozen SigLIP (0.23% of parameters) raises **KonIQ-10k SROCC from 0.887 to 0.951**.
- On synthetic benchmarks, **image-level splitting makes every dataset look uniformly easy (0.95–0.97)**,
  masking true difficulty that only **reference-level splitting** reveals (0.51–0.91).
- The inflation from content overlap (0.06–0.44) is **not explained by the number of reference images**
  (rank correlation ρ = −0.05).
- Across six datasets, **adaptation gain tracks frozen-feature weakness**: large where frozen features
  are weak, negligible where they are already strong.

---

## Results

### Adaptation across six datasets (reference-level splitting)

| Dataset | Frozen SROCC | + LoRA SROCC | Δ |
|---|---|---|---|
| TID2013 | 0.514 | 0.871 | +0.357 |
| PIPAL | 0.576 | 0.707 | +0.131 |
| KADID-10k | 0.787 | 0.927 | +0.141 |
| KonIQ-10k | 0.887 | 0.951 | +0.064 |
| LIVE-MD | 0.904 | 0.893 | −0.011 |
| CSIQ | 0.912 | 0.927 | +0.015 |

### Content overlap inflates frozen-feature performance (five synthetic datasets)

| Dataset | Refs | Image-level | Reference-level | Gap |
|---|---|---|---|---|
| CSIQ | 30 | 0.969 | 0.912 | 0.057 |
| LIVE-MD | 30 | 0.965 | 0.904 | 0.061 |
| KADID-10k | 81 | 0.955 | 0.787 | 0.168 |
| PIPAL | 200 | 0.793 | 0.576 | 0.217 |
| TID2013 | 25 | 0.950 | 0.514 | 0.436 |

> **Note on KADID-10k.** Under the conventional image-level protocol our frozen system reaches a high
> SROCC on KADID-10k, but as the table above shows, this protocol inflates scores through content
> overlap between train and test partitions. We therefore base our analysis on the reference-level
> protocol and report image-level numbers only for comparison with prior work.

---

## Installation

```bash
git clone https://github.com/bishr-omer/multimodal-biqa.git
cd multimodal-biqa
pip install -r requirements.txt
```

> Tested on Python 3.10. A GPU is required for one-time feature extraction and for LoRA adaptation
> (which backpropagates through the SigLIP backbone). Frozen-head training runs on CPU.
> On Google Colab, install `peft==0.11.1` to avoid a torchao version conflict.

---

## Usage

### 1. Extract frozen features (GPU, one-time)

```bash
python extract_features.py --dataset koniq --image_dir /path/to/koniq --out features/koniq.npz
```

Features are cached to `.npz` with checkpointing, so an interrupted run resumes where it stopped.

### 2. Train the frozen regression head

```bash
python train_head.py --dataset koniq --features features/koniq.npz
```

### 3. Evaluate under both splitting protocols (synthetic datasets)

```bash
python evaluate_protocols.py --dataset kadid --features features/kadid.npz
# reports image-level and reference-level SROCC and the gap
```

### 4. LoRA adaptation (GPU; requires raw images, not cached features)

```bash
python lora_adapt.py --dataset tid2013 --image_dir /path/to/tid2013 --split reference
```

### 5. Reproduce the figures

```bash
python make_figures.py   # writes fig3_leakage.png, fig4_adaptation.png, fig5_curves.png
```

---

## Datasets

| Dataset | Type | Refs | Source |
|---|---|---|---|
| KonIQ-10k | Authentic | — | [link](https://database.mmsp-kn.de/koniq-10k-database.html) |
| LIVE-itW | Authentic | — | [link](https://live.ece.utexas.edu/research/ChallengeDB/index.html) |
| KADID-10k | Synthetic | 81 | [link](http://database.mmsp-kn.de/kadid-10k-database.html) |
| TID2013 | Synthetic | 25 | [link](https://www.ponomarenko.info/tid2013.htm) |
| CSIQ | Synthetic | 30 | [link](https://s2.smu.edu/~eclarson/csiq.html) |
| LIVE-MD | Synthetic | 30 | [link](https://live.ece.utexas.edu/research/Quality/live_multidistortedimage.html) |
| PIPAL | Synthetic | 200 | [link](https://github.com/HaomingCai/PIPAL-dataset) |

Synthetic datasets (with meta-info containing reference identities and MOS) are also available through
the [IQA-PyTorch toolbox datasets](https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets).

---

## Citation

```bibtex
@article{adam2026lora,
  title   = {Low-Rank Adaptation of Frozen Vision-Language Models
             for Blind Image Quality Assessment},
  author  = {Adam, Bishr Omer Abdelrahman and Li, Xu},
  year    = {2026}
}
```

---

## Acknowledgements

VLM feature extraction uses [OpenCLIP](https://github.com/mlfoundations/open_clip) and
[Hugging Face Transformers](https://github.com/huggingface/transformers). LoRA adaptation uses
[PEFT](https://github.com/huggingface/peft).
