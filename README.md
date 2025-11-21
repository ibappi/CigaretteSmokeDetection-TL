# 📘 Enhancing Vision-Based Cigarette Smoke Detection in Smart Vehicles
This repository provides the implementation of our model for vision-based cigarette smoke detection in smart vehicle environments, as proposed in the following peer-reviewed publication:

---

## 🔍 Overview

Our model addresses these challenges through:
- Transfer Learning using VGG19
- Fine-tuning for binary classification (smoke / non-smoke)
- Spatial Attention Module for improved region focus
- Grad-CAM & Attention Map visualization for interpretability
- Lightweight architecture suitable for real-time smart vehicle systems

Bappi, MD Ilias; Jin, Hyeonseok; Kim, Kyungbaek (2025).
Enhancing Vision-Based Cigarette Smoke Detection in Smart Vehicles by Transfer Learning.
Journal of Digital Contents Society, 26(4), 1041–1057.

---

## 📁 Repository Structure

```
CigaretteSmokeDetection-TL/
│
├── README.md
│
├── src/
│   ├── models/
│   │   ├── vgg19_baseline.py
│   │   ├── vgg19_spatial_attention.py
│   │   └── attention_module.py
│   │
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── preprocess.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── dataset/
│   ├── README_DATASET.txt           ← Explains dataset structure only
│   └── sample_paths.txt             ← Sample example of image paths
│
├── requirements.txt
│
│
└── .gitignore
```

## 📄 Citation

```bibtex
@article{bappi2025enhancing,
  title={Enhancing Vision-Based Cigarette Smoke Detection in Smart Vehicles by Transfer Learning},
  author={Bappi, MD Ilias and Jin, Hyeonseok and Kim, Kyungbaek},
  journal={디지털콘텐츠학회논문지},
  volume={26},
  number={4},
  pages={1041--1057},
  year={2025}
}


