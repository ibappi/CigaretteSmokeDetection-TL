📘 Enhancing Vision-Based Cigarette Smoke Detection in Smart Vehicles
Transfer Learning with VGG19 (Official Code)

This repository contains the official implementation of the model proposed in the following peer-reviewed publication:

Bappi, MD Ilias; Jin, Hyeonseok; Kim, Kyungbaek (2025).
Enhancing Vision-Based Cigarette Smoke Detection in Smart Vehicles by Transfer Learning.
Journal of Digital Contents Society, 26(4), 1041–1057.

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
│   │   ├── visualization.py
│   │   └── metrics.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── dataset/
│   ├── README_DATASET.txt           ← Explains dataset structure only
│   └── sample_paths.txt             ← Sample example of image paths
│
├── experiments/
│   ├── logs/
│   ├── results/
│   └── attention_maps/              ← Grad-CAM & spatial attention outputs
│
├── requirements.txt
│
├── LICENSE
│
└── .gitignore
This work focuses on enhancing real-time cigarette smoke detection in smart vehicles using a transfer-learning approach based on VGG19.
Our contributions include:

Fine-tuned VGG19 model for binary classification (smoke vs. non-smoke)
Optional Spatial Attention Module (CBAM-like) for improved localization
Robust feature extraction for dense, sparse, and ambiguous smoke patterns
Visualization tools including Grad-CAM and spatial attention heatmaps
This codebase is designed for reproducibility and for extending the detection framework in smart vehicle applications.

