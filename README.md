# Q-DISTILL: Bridging the Multimodal Information Gap in Chest X-ray Triage via Self-Supervised Q-Former Distillation

While vision-language models excel in medical imaging, their reliance on matched reports limits utility for image-only triage. We identify a fundamental multimodal information gap in distilling these models to image-only students, where text-dependent features are irrecoverable. We propose **Q-DISTILL**, a self-supervised framework that bridges this gap using MedGemma-generated pseudo-reports as textual proxies. Our approach reveals that foundation models can serve not only as generators but as **knowledge amplifiers**---providing both weak supervision and feature proxies, boosting accuracy from 76.5% to 89.1% while compressing 27B-parameter reasoning into deployable architectures. 

## Overview

**Q-DISTILL** trains a lightweight, image-only model to predict radiology report complexity by distilling knowledge from a multimodal teacher model.

## Architecture

### Stage 1: Multimodal Teacher
- **Vision**: CheXzero (medical CLIP)
- **Text**: Simple Text Encoder
- **Fusion**: Q-Former with learnable queries
- **Training**: Contrastive + Image-Text Matching + Focal Classification Loss

![Teacher Model Architecture](model_images/Model%20arc.001.jpeg)

### Stage 2: Lightweight Student
- **Vision**: CheXzero (frozen)
- **Fusion**: Lightweight Q-Former
- **Training**: Knowledge distillation from frozen teacher

<p align="center">
  <img src="model_images/Model%20arc.002.jpeg" 
       alt="Student Model Architecture" 
       width="600">
</p>

## Installation

```bash
# Clone repository
git clone https://github.com/Yangxinyee/Q-DISTILL.git
cd Q-DISTILL

# Create environment
conda create -n qdistill python=3.10 -y
conda activate qdistill

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Download chexzero checkpoint from:

https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno

Store it in the `checkpt/` directory as `chexzero_best.pt`.

Organize your data as (matching the current `data/` directory in this repo):
```
data/
├── images/                     # CXR images (.png)
├── reports/                    # Ground-truth radiology reports (.txt)
├── metadata.csv                # Main index used by teacher/student
├── medgemma_reports/           # MedGemma-generated reports + index
│   ├── metadata.csv            # study_id, medgemma_report_path, split, weak_label, discrete_label
│   └── *_medgemma_report.txt   # Generated report text files
└── priority_scores/            # Optional: per-study priority jsons (if available)
    └── *_priority.json
```

Notes:
- `metadata.csv` must contain at least: `study_id, image_path, report_path, split, priority_label`
  - `image_path` / `report_path` are **paths relative to `--root_dir`** (typically `data/`), e.g. `images/xxx.png`, `reports/xxx.txt`
- `data/medgemma_reports/metadata.csv` must contain at least: `study_id, medgemma_report_path, split`
  - `medgemma_report_path` is **relative to `--medgemma_report_dir`** (typically `data/medgemma_reports/`), e.g. `60244513_medgemma_report.txt`
- `split` should be one of: `train`, `validation`, `test`

### 2. Build Vocabulary

```bash
python build_vocabulary.py \
    --metadata_path data/metadata.csv \
    --root_dir data \
    --output_path vocab_test/vocabulary.json
```

### 3. Train Teacher Model

PS: if your torch.cuda.is_available() is True, then you can use AMP. Otherwise please delete `--use_amp`

```bash
python train_teacher.py \
    --metadata_path data/metadata.csv \
    --root_dir data \
    --vocab_path vocab/vocabulary.json \
    --chexzero_path checkpt/chexzero_best.pt \
    --batch_size 64 \
    --epochs 150 \
    --output_dir outputs/teacher \
    --use_amp
```

### 4. Train Student Model

PS: if your torch.cuda.is_available() is True, then you can use AMP. Otherwise please delete `--use_amp`

```bash
python train_student.py \
    --teacher_checkpoint outputs/teacher/checkpoint_best.pt \
    --chexzero_path checkpt/chexzero_best.pt \
    --csv_path data/metadata.csv \
    --medgemma_metadata_path data/medgemma_reports/metadata.csv \
    --root_dir data \
    --medgemma_report_dir data/medgemma_reports \
    --vocab_path vocab/vocabulary.json \
    --batch_size 32 \
    --epochs 50 \
    --distill_target bi \
    --output_dir outputs/student \
    --use_amp
```

### 4.5 Train CNN Baseline (Optional)

PS: AMP is enabled by default. If your torch.cuda.is_available() is False, please add `--no_amp`.

```bash
# Standard CNN baseline (train from scratch)
python train_cnn_baseline.py \
    --csv_path data/metadata.csv \
    --root_dir data \
    --batch_size 32 \
    --epochs 50  \
    --output_dir outputs/cnn_baseline

# CNN baseline with CheXzero as feature extractor
python train_cnn_baseline.py \
    --csv_path data/metadata.csv \
    --root_dir data \
    --batch_size 32 \
    --epochs 50 \
    --use_chexzero \
    --chexzero_path checkpt/chexzero_best.pt \
    --output_dir outputs/cnn_baseline_chexzero
```

### 5. Evaluate

PS: If you only want to plot the training curves, please add `--plot_curves`

```bash
# Teacher evaluation
python evaluate_teacher.py \
    --checkpoint outputs/teacher/checkpoint_best.pt \
    --metadata_path data/metadata.csv \
    --root_dir data \
    --chexzero_path checkpt/chexzero_best.pt \
    --split test

# Student evaluation
python evaluate_student.py \
    --student_checkpoint outputs/student/checkpoint_best.pt \
    --teacher_checkpoint outputs/teacher/checkpoint_best.pt \
    --chexzero_path checkpt/chexzero_best.pt \
    --csv_path data/metadata.csv \
    --medgemma_metadata_path data/medgemma_reports/metadata.csv \
    --vocab_path vocab/vocabulary.json \
    --split test

# CNN baseline evaluation
python eval_cnn_baseline.py \
    --checkpoint outputs/cnn_baseline/checkpoint_best.pt \
    --csv_path data/metadata.csv \
    --root_dir data \
    --split test
```

## Project Structure

```
Q-DISTILL/
├── model/
│   ├── dataset.py              # Datasets + collate functions
│   ├── teacher.py              # Teacher model (Q-Former + text encoder)
│   ├── student.py              # Student model (lightweight Q-Former)
│   ├── cnn_baseline.py         # CNN baseline model
│   ├── losses.py               # Distillation + classification losses
│   ├── chexzero_vision_encoder.py  # CheXzero wrapper
│   ├── clip_model.py           # CLIP implementation
│   └── __init__.py
│
├── build_vocabulary.py         # Vocabulary builder
├── generate_medgemma_reports_multigpu.py  # Generate MedGemma reports (multi-GPU)
├── score_report_complexity_multigpu.py    # Score report complexity (multi-GPU)
│
├── train_teacher.py            # Stage 1: train teacher
├── train_student.py            # Stage 2: train student (distillation)
├── train_cnn_baseline.py       # Train CNN baseline
│
├── evaluate_teacher.py         # Teacher evaluation
├── evaluate_student.py         # Student evaluation
├── eval_cnn_baseline.py        # CNN baseline evaluation
│
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@misc{qdistill2025,
  author = {Xinye Yang},
  title = {Q-DISTILL: Knowledge Distillation for Chest X-Ray Diagnostic Complexity},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Yangxinyee/Q-DISTILL}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [CheXzero](https://github.com/rajpurkarlab/CheXzero) for medical vision encoder
- [BLIP-2](https://github.com/salesforce/LAVIS) for Q-Former architecture inspiration
