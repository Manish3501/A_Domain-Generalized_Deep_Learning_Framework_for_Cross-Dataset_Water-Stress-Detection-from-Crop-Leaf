# Domain-Adversarial Deep Learning for Cross-Dataset Water Stress Detection from Crop Leaf Images

## Overview

This project presents a domain-adversarial deep learning framework for
detecting plant water stress from crop leaf images across three
independently collected datasets. The core challenge addressed is
cross-dataset generalization — existing models train and test on a
single dataset and fail when applied to images from different crops,
sensors, or environments. This work treats three datasets as three
distinct domains and uses a Domain-Adversarial Neural Network (DANN)
with a Gradient Reversal Layer (GRL) to learn features that detect
water stress without relying on dataset-specific shortcuts.

At inference, the system accepts a single leaf image from any source
and produces one binary stress prediction without requiring any
knowledge of which dataset the image originates from.

The task is binary classification:
- **0 = Non-Stress**
- **1 = Stress**

---

## Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Ensemble | 71.43% | 0.7031 | 0.9910 | 0.8226 | 0.7037 |
| Improved Ensemble | 75.40% | 0.7331 | 0.9937 | 0.8437 | 0.9446 |
| Feature Fusion† | — | — | — | — | — |
| **Proposed DANN** | **97.32%** | **0.9847** | **0.9751** | **0.9799** | **0.9950** |

> † Feature Fusion was evaluated on only 48 of 9,269 test samples due
> to batch alignment constraints (zip() with drop_last=True, shortest
> loader = Maize with 60 test images). Results are not directly
> comparable and are excluded from the table.

### Per-Domain Results (Proposed DANN)

| Domain | Samples | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Tomato (D1) | 3,144 | 99.97% | 0.9997 | 0.9997 | 0.9997 |
| Maize (D2)‡ | 60 | 91.67% | 0.9375 | 0.9167 | 0.9202 |
| Maize2 (D3) | 6,065 | 96.01% | 0.9604 | 0.9601 | 0.9602 |
| **Combined** | **9,269** | **97.32%** | **0.9847** | **0.9751** | **0.9799** |

> ‡ Maize results are based on only 60 test samples. Interpret with
> caution — results are indicative rather than statistically stable.

---

## Datasets

Three independently sourced leaf image datasets are used as three
separate domains. All images are resized to 224×224 and normalized
using ImageNet statistics (mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]).

| Domain | Crop | Total | Train | Val | Test | Format |
|---|---|---|---|---|---|---|
| D1 — Tomato | Tomato | 20,955 | 14,668 | 3,143 | 3,144 | RGB images |
| D2 — Maize | Maize | 400 | 280 | 60 | 60 | RGB images |
| D3 — Maize2 | Maize | 18,040 | 9,580 | 2,395 | 6,065 | NumPy arrays (48×48×3) |
| **Combined** | | **39,395** | **24,528** | **5,598** | **9,269** | |

The datasets differ in crop type, image resolution, background,
imaging conditions, and data collection procedure, making
cross-dataset generalization a genuine and non-trivial challenge.

> **Data Access:** The datasets were obtained directly from the
> respective authors on request and are not included in this
> repository. Please contact the original dataset authors to request
> access before running this code.

---

## Project Structure

```text
leaf-based-water-stress-detection/
│
├── src/
│   ├── data_loader.py       # Loads and preprocesses all three datasets.
│   │                        # Assigns binary stress labels (0/1) and
│   │                        # domain IDs (0=Tomato, 1=Maize, 2=Maize2).
│   │                        # Handles both image files (Tomato, Maize)
│   │                        # and NumPy arrays (Maize2). Applies
│   │                        # stratified 70/15/15 train/val/test splits.
│   │
│   ├── models.py            # All model architectures:
│   │                        #   StressClassifier — single-domain CNN
│   │                        #     (MobileNetV2 + classifier head)
│   │                        #   FusionModel — three-stream MobileNetV2
│   │                        #     (concatenates 3x1280=3840 features)
│   │                        #   GradientReversalLayer — custom autograd
│   │                        #     function (negates gradient x -lambda)
│   │                        #   DomainGeneralizationModel — DANN
│   │                        #     (shared backbone + stress head + GRL
│   │                        #     + domain head)
│   │
│   ├── train.py             # Training functions for all four approaches:
│   │                        #   train_model — trains a single CNN
│   │                        #     (used for ensemble sub-models)
│   │                        #   train_fusion_model — trains the three-
│   │                        #     stream fusion model with zip() loaders
│   │                        #   train_domain_generalization — trains the
│   │                        #     DANN with lambda annealing, dual loss,
│   │                        #     GRL, ReduceLROnPlateau scheduler, and
│   │                        #     domain classifier accuracy tracking
│   │
│   ├── evaluate.py          # Evaluation functions:
│   │                        #   evaluate_model — single model accuracy
│   │                        #   evaluate_ensemble — uniform avg ensemble
│   │                        #   evaluate_improved_ensemble — weighted avg
│   │                        #   evaluate_fusion_model — three-stream eval
│   │                        #   evaluate_dg_model — DANN eval with per-
│   │                        #     domain accuracy breakdown
│   │                        #   collect_predictions_single — collects
│   │                        #     preds + probs for ROC curve
│   │                        #   collect_predictions_fusion — fusion preds
│   │                        #   compute_all_metrics — accuracy, precision,
│   │                        #     recall, F1, ROC-AUC, classification report
│   │                        #   plot_confusion_matrix — heatmap with counts
│   │                        #     and percentages
│   │                        #   plot_roc_curves — all models on one axes
│   │
│   ├── transforms.py        # Image transform pipelines:
│   │                        #   transform — training augmentation
│   │                        #     (RandomHorizontalFlip, RandomVerticalFlip,
│   │                        #     RandomRotation ±20°, RandomResizedCrop,
│   │                        #     ColorJitter, ToTensor, Normalize)
│   │                        #   val_transform / dg_val_transform —
│   │                        #     evaluation only (Resize, ToTensor,
│   │                        #     Normalize, no augmentation)
│   │
│   └── gradcam.py           # Grad-CAM implementation for model
│                            # interpretability. Generates activation
│                            # heatmaps overlaid on leaf images to show
│                            # which regions drive stress predictions.
│                            # Applied across all three domains.
│
├── Notebooks/
│   ├── Ensemble_Learning.ipynb      # Trains and saves the plain ensemble
│   │                                # (5 epochs per model, uniform average)
│   │                                # and the improved ensemble (15 epochs,
│   │                                # augmentation, weighted 0.5/0.3/0.2).
│   │                                # Saves models to saved_models/.
│   │
│   ├── Fusion_model.ipynb           # Trains and saves the three-stream
│   │                                # feature fusion model (10 epochs).
│   │                                # Note: evaluation limited to 48 samples
│   │                                # due to zip()+drop_last constraints.
│   │
│   └── Domain_generalization.ipynb  # Trains and saves the proposed DANN
│                                    # (25 epochs, Adam lr=1e-4,
│                                    # weight_decay=1e-4, batch=32,
│                                    # ReduceLROnPlateau patience=3).
│                                    # Tracks domain classifier accuracy
│                                    # per epoch to verify GRL is working.
│
├── Evaluation/
│   └── all_models_evaluation.ipynb  # Loads all saved models and runs
│                                    # unified evaluation across the full
│                                    # 9,269-image combined test set.
│                                    # Produces: accuracy, precision,
│                                    # recall, F1, ROC-AUC, confusion
│                                    # matrices, per-domain metrics
│                                    # (precision/recall/F1 per domain),
│                                    # and combined ROC curve plot.
│
└── README.md
```

---

## Proposed Architecture (DANN)

The proposed model has three components that share a single forward pass:

### 1. Shared Backbone
MobileNetV2 pretrained on ImageNet. Processes every input image
through 18 convolutional layers. Global average pooling produces a
fixed-length 1280-dimensional feature vector.

### 2. Stress Classifier Head (retained at inference)
```
Linear(1280 → 512) → ReLU → Dropout(0.5)
Linear(512  → 128) → ReLU → Dropout(0.3)
Linear(128  → 2)
```
Loss: CrossEntropyLoss on binary stress label (0 or 1).

### 3. Domain Classifier Head with GRL (discarded at inference)
```
GRL(λ) → Linear(1280 → 256) → ReLU → Dropout(0.4) → Linear(256 → 3)
```
Loss: CrossEntropyLoss on domain label (0=Tomato, 1=Maize, 2=Maize2).

### Training Objective
```
Total Loss = Stress Loss + λ × Domain Loss
```

The GRL negates the domain gradient during back-propagation,
forcing the backbone to learn features that cannot identify the
source dataset while still detecting stress correctly.

### Lambda Annealing Schedule
```
λ = 2 / (1 + exp(-10 × p)) - 1,  where p = epoch / total_epochs
```
λ starts at 0 (epoch 0) and reaches ~0.96 at epoch 24, gradually
increasing adversarial pressure as training progresses.

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn opencv-python
```

### 2. Prepare datasets
Place your datasets in the expected locations and update the file
paths in `src/data_loader.py`. The three datasets require:
- Tomato: folder of RGB image files organised by field-capacity level
- Maize: folder of RGB image files organised by irrigation category
- Maize2: NumPy arrays (trainData.npy, valData.npy, testData.npy)
  with corresponding label files

### 3. Run notebooks in this order
```
1. Notebooks/Ensemble_Learning.ipynb
2. Notebooks/Fusion_model.ipynb
3. Notebooks/Domain_generalization.ipynb
4. Evaluation/all_models_evaluation.ipynb
```

Each training notebook saves its model weights to a `saved_models/`
folder. The evaluation notebook loads all saved models, so the three
training notebooks must be run first.

---

## Training Configuration

| Setting | Ensemble | Improved Ensemble | Fusion | DANN |
|---|---|---|---|---|
| Epochs | 5 | 15 | 10 | 25 |
| Optimiser | Adam | Adam | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Weight decay | — | — | — | 1e-4 |
| Batch size | 32 | 32 | 16 | 32 |
| Scheduler | None | None | None | ReduceLROnPlateau (patience=3, factor=0.5) |
| Augmentation | None | Yes | None | Yes |
| Prediction | Uniform avg | Weighted 0.5/0.3/0.2 | Concatenated features | Single image |

### Augmentation (applied during DANN and Improved Ensemble training)
- RandomHorizontalFlip (p=0.5)
- RandomVerticalFlip (p=0.3)
- RandomRotation (±20°)
- RandomResizedCrop (224×224, scale=0.8–1.0)
- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

---

## Requirements

```
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
opencv-python
```

Python 3.8 or higher recommended. Tested on Apple Silicon (MPS backend).
CUDA-compatible GPU also supported — device selection is automatic.

---

## Domain Classifier Accuracy (GRL Verification)

To verify that the GRL is functioning correctly, domain classifier
accuracy is tracked during DANN training. A working GRL should cause
the domain classifier accuracy to fall from a high initial value
toward the random-chance baseline of 33.3% (1/3 for three domains)
as lambda increases.

In this experiment the domain classifier accuracy fell to **59.77%**
by the final epoch (epoch 25, λ=0.96). While it did not reach the
theoretical chance level of 33.3%, this is expected given the extreme
visual heterogeneity between the three datasets — a high-resolution
tomato photograph, a field maize image, and a 48×48 pixel array will
always retain some domain-identifiable characteristics. The partial
suppression achieved was sufficient for the backbone to learn
stress features that generalize across all three domains, as
confirmed by the 97.32% combined test accuracy.

---

## Author

**Manish Kailas Chaudhari**
MSc Computer Science (Data Analytics)
University of Galway, Ireland
m.chaudhari2@universityofgalway.ie

---

## Acknowledgements

The datasets used in this project were obtained from the respective
authors on request. The project supervisor provided guidance and
feedback throughout the research process.

---


