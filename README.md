# A Domain-Generalized Deep Learning Framework for Cross-Dataset Water Stress Detection from Crop Leaf Images

This project presents a deep learning framework for detecting plant water stress from leaf images across multiple datasets.

The proposed approach uses Domain-Adversarial Neural Networks (DANN) with a MobileNetV2 backbone to learn features that are useful for water-stress classification while reducing dataset-specific differences between domains.

Three independently collected datasets are used:

- Tomato
- Maize
- Maize2

The task is formulated as binary classification:

- 0 = Non-Stress
- 1 = Stress

---

## Project Overview

The project compares several approaches for leaf-based water stress detection:

1. Ensemble
2. Improved Ensemble
3. Feature Fusion
4. Proposed Domain Generalization (DANN)

The main proposed model uses:

- MobileNetV2 as the feature extractor
- Stress classifier for water-stress prediction
- Domain classifier for domain prediction
- Gradient Reversal Layer (GRL) for domain-adversarial training

The aim is to learn features that generalize across different datasets and imaging conditions.

---

## Dataset

Three datasets are used as separate domains.

### Domain 1: Tomato
### Domain 2: Maize
### Domain 3: Maize2

The datasets differ in:

- Crop type
- Image resolution
- Background
- Imaging conditions
- Dataset collection procedure

All images are converted to the required input format and resized to 224 × 224 for the MobileNetV2 model.

---

## Project Structure

```text
leaf-based-water-stress-detection/
│
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── transforms.py
│   └── gradcam.py
│
├── Notebooks/
│   ├── Domain_Genaralization.ipynb
│   ├── Ensemble_Learning.ipynb
│   ├── Fusion_model.ipynb
│
├── Evaluation/
│   └── all_models_evaluation.ipynb
|
├── README.md
