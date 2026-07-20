<div align="center">

# Parameter-Efficient Adaptation of a Multi-Stream Vision-Language Framework<br>for Blind Image Quality Assessment

**Bishr Omer Adam · Xu Li**

Northwestern Polytechnical University, Xi'an, China

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

This repository contains the code for our study of **parameter-efficient adaptation of a multi-stream
vision-language framework for blind image quality assessment (BIQA)**, evaluated across six datasets
under both image-level and reference-level splitting protocols, with epoch selection performed on a
held-out validation split and results reported as mean ± std over multiple reference-level splits.

The method has two parts:

1. **An efficient frozen-feature framework.** A 138-dimensional natural-scene-statistics (NSS)
   descriptor is fused with frozen **SigLIP** and **CLIP-H** embeddings through a lightweight MLP
   regression head trained on cached features. No end-to-end fine-tuning of the VLM backbones is
   required, and the head trains quickly even on CPU.

2. **Parameter-efficient backbone adaptation.** The SigLIP stream of this framework is then adapted
   with **Low-Rank Adaptation (LoRA)**, training only **0.23%** of its parameters. We further compare
   LoRA against **DoRA** and **AdaLoRA** under an identical protocol.

| Stream | Backbone | Output |
|---|---|---|
| **NSS** | Spatial + color/frequency statistics | 138D |
| **SigLIP** | ViT-SO400M-14-SigLIP-384 | 1152D → 256D (PCA) |
| **CLIP-H** | ViT-H-14 LAION-2B | 1024D → 256D (PCA) |

---

## Key Findings

- LoRA adaptation of frozen SigLIP (0.23% of parameters) raises **KonIQ-10k SROCC from 0.887 to
  0.948 ± 0.004**.
- On synthetic benchmarks, **image-level splitting makes every dataset look uniformly easy (0.79–0.97)**,
  masking true difficulty that only **reference-level splitting** reveals (0.51–0.91).
- The inflation from content overlap (0.06–0.44) is **not explained by the number of reference images**
  (rank correlation ρ = −0.05).
- Across six datasets, **adaptation gain tracks frozen-feature weakness**: large where frozen features
  are weak, negligible where they are already strong.
- Compared against **DoRA** and **AdaLoRA** under an identical protocol, vanilla **LoRA** matches or
  beats both alternatives while training fewer parameters and in less time.

---

## Results

All adaptation results use a three-way reference-level split (train / validation / test); the epoch is
selected on the validation partition, the test partition is scored once, and each entry is the mean ± std
over $n$ independent splits.

### Adaptation across six datasets (reference-level splitting)

| Dataset | Frozen SROCC | + LoRA SROCC | Δ | $n$ |
|---|---|---|---|---|
| TID2013 | 0.514 | 0.842 ± 0.071 | +0.328 | 5 |
| PIPAL | 0.576 | 0.659 ± 0.010 | +0.083 | 2 |
| KADID-10k | 0.787 | 0.921 ± 0.009 | +0.134 | 3 |
| KonIQ-10k | 0.887 | 0.948 ± 0.004 | +0.061 | 3 |
| LIVE-MD | 0.904 | 0.911 ± 0.027 | +0.007 | 5 |
| CSIQ | 0.912 | 0.946 ± 0.013 | +0.034 | 5 |

### Content overlap inflates frozen-feature performance (five synthetic datasets)

| Dataset | Refs | Image-level | Reference-level | Gap |
|---|---|---|---|---|
| CSIQ | 30 | 0.969 | 0.912 | 0.057 |
| LIVE-MD | 30 | 0.965 | 0.904 | 0.061 |
| KADID-10k | 81 | 0.955 | 0.787 | 0.168 |
| PIPAL | 200 | 0.793 | 0.576 | 0.217 |
| TID2013 | 25 | 0.950 | 0.514 | 0.436 |

### Comparison of parameter-efficient adaptation methods (TID2013)

| Method | SROCC | Trainable | Rel. time |
|---|---|---|---|
| LoRA | 0.842 ± 0.071 | 0.232% | 1.0× |
| DoRA | 0.819 ± 0.083 | 0.246% | 4.3× |
| AdaLoRA | 0.782 ± 0.084 | 0.348% | 3.2× |

Neither DoRA nor AdaLoRA improves on vanilla LoRA under this protocol, so we retain the simpler method.

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

> Tested on Python 3.10. A GPU is required for one-time feature extraction and for LoRA/DoRA/AdaLoRA
> adaptation (which backpropagates through the SigLIP backbone). Frozen-head training runs on CPU.
> On Kaggle/Colab, install `peft==0.11.1` and uninstall `torchao` to avoid a version conflict.

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

Uses a three-way reference-level split with validation-based epoch selection and multi-seed averaging:

```bash
python lora_adapt_fixed.py --dataset tid2013 --meta_dir meta_info \
    --img_root /path/to/tid2013/distorted_images --seeds 0 1 2 3 4 --epochs 6
```

### 5. PEFT comparison (LoRA vs. DoRA vs. AdaLoRA)

```bash
python adapt_variants.py --dataset tid2013 --method dora \
    --meta_dir meta_info --img_root /path/to/tid2013/distorted_images --seeds 0 1 2 3 4 --epochs 6
```

### 6. Reproduce the figures

```bash
python make_figures_v2.py   # writes Figure_2.png (leakage), Figure_3.png (adaptation), Figure_4.png (curves)
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
@article{adam2026peft,
  title   = {Parameter-Efficient Adaptation of a Multi-Stream Vision-Language
             Framework for Blind Image Quality Assessment},
  author  = {Adam, Bishr Omer and Li, Xu},
  year    = {2026}
}
```

---

## Acknowledgements

VLM feature extraction uses [OpenCLIP](https://github.com/mlfoundations/open_clip) and
[Hugging Face Transformers](https://github.com/huggingface/transformers). LoRA/DoRA/AdaLoRA adaptation
uses [PEFT](https://github.com/huggingface/peft).
