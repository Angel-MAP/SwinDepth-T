# SwinDepth-T

Official repository for **"SwinDepth-T: Revolutionizing Monocular Depth Estimation with Hierarchical Transformer-Based Multi-Scale Feature Fusion"**.

## Overview
SwinDepth-T is a hierarchical Transformer-based model designed for monocular depth estimation. It combines:
- Shifted Window Attention
- Multi-Scale Feature Fusion Attention Module (MSFA-Net)
- Lightweight and scalable encoder-decoder architecture

## Highlights
- Linear computational complexity with respect to image size
- Supports KITTI and NYU Depth v2 datasets
- Compact, Simple, and Standard variants (31M–93M params)
- State-of-the-art accuracy with reduced inference time

##  Structure
- `train.py`: Training script
- `test.py`: Evaluation script
- `models/`: Model definitions
- `configs/`: Hyperparameters
- `weights/`: Pretrained models
- `utils/`: Helper functions and evaluation metrics

## Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/Angel-MAP/SwinDepth-T.git
cd SwinDepth-T

2. Install dependencies
pip install -r requirements.txt

3. Run training
python train.py --config configs/swin_tiny.yaml

4. Run evaluation
python test.py --weights weights/swin_tiny.pth


Pre-trained Models
Download links (hosted here in weights/):
swin_tiny.pth
swin_simple.pth
swin_standard.pth

Evaluation Results
Results on KITTI and NYU datasets included in /logs/benchmark_results.

Benchmark Results

The following are the benchmark evaluation results for **SwinDepth-T** variants on the KITTI and NYU Depth v2 datasets.  
Logs are available in both `.txt` (human-readable) and `.csv` (structured) formats.

### KITTI
- Swin-Tiny: [TXT](logs/benchmark_results/kitti_swin_tiny.txt) | [CSV](logs/benchmark_results/kitti_swin_tiny.csv)
- Swin-Simple: [TXT](logs/benchmark_results/kitti_swin_simple.txt) | [CSV](logs/benchmark_results/kitti_swin_simple.csv)
- Swin-Standard: [TXT](logs/benchmark_results/kitti_swin_standard.txt) | [CSV](logs/benchmark_results/kitti_swin_standard.csv)

### NYU Depth v2
- Swin-Tiny: [TXT](logs/benchmark_results/nyu_swin_tiny.txt) | [CSV](logs/benchmark_results/nyu_swin_tiny.csv)
- Swin-Simple: [TXT](logs/benchmark_results/nyu_swin_simple.txt) | [CSV](logs/benchmark_results/nyu_swin_simple.csv)
- Swin-Standard: [TXT](logs/benchmark_results/nyu_swin_standard.txt) | [CSV](logs/benchmark_results/nyu_swin_standard.csv)

Pretrained Weights

We provide pretrained model weights for all **SwinDepth-T** variants.  
These placeholders replaced with the actual trained weights .

### KITTI-trained Models
- Swin-Tiny: (weights/swin_tiny.pth)
- Swin-Simple: (weights/swin_simple.pth)
- Swin-Standard: (weights/swin_standard.pth)

Citation
@article{ponrani2025swindeptht,
  title={SwinDepth-T: Revolutionizing Monocular Depth Estimation with Hierarchical Transformer-Based Multi-Scale Feature Fusion},
  author={Ponrani, M. Angelin and Ezhilarasi, P. and Rajeshkannan, S.},
  journal={Journal of Electronic Imaging},
  year={2025}
}







