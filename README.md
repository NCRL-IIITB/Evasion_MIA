<div align="center">

# 🛡️ Robustness-Privacy Tradeoff in Adversarially Trained Medical Image Classifiers

### Demonstrating how defending against evasion attacks opens vulnerability to membership inference

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: NIH Chest X-ray](https://img.shields.io/badge/Dataset-NIH%20Chest%20X--ray-orange)](https://nihcc.app.box.com/v/ChestXray-NIHCC)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Pipeline Execution](#pipeline-execution)
- [Attack Methodology](#attack-methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Configuration and Hyperparameters](#configuration-and-hyperparameters)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## 🔬 Overview

This project investigates a fundamental tension in machine learning security: **defending a model against one class of attack can make it vulnerable to another**.

We study this tradeoff in the context of **medical image classification**, specifically using NIH Chest X-rays across 15 different pathologies. We focus on two types of attacks:

| Attack Type | Goal | Threat |
|:---|:---|:---|
| **Evasion Attack** (FGSM) | Fool the model at inference time by adding imperceptible noise to inputs | Misdiagnosis of patients |
| **Membership Inference Attack** (MIA) | Determine whether a specific patient's data was used to train the model | Privacy violation (HIPAA, GDPR) |

**The core finding:** Adversarial training, which is the standard defense against evasion attacks, causes what is known as **robust overfitting**. This means the model memorizes adversarial patterns specific to its training data. Our novel **fluctuation-based MIA** exploits this memorization to infer membership with **67.4% accuracy** (compared to 50.2% on the undefended baseline), all while maintaining a complete **black-box** threat model.

### The Robustness-Privacy Tradeoff

This project clearly demonstrates how defending against evasion opens up new vulnerabilities:
- **Baseline Model:** Vulnerable to evasion attacks (26.1% flip rate) but safe from privacy leaks (50.2% MIA accuracy).
- **Adversarial Model:** Robust against evasion (3.9% flip rate) but highly vulnerable to privacy leaks (67.4% MIA accuracy).

---

## 🎯 Key Findings

| Metric | Baseline Model | Adversarial Model |
|:---|:---:|:---:|
| **Clean AUC** | 0.828 | 0.752 |
| **FGSM Flip Rate** (ε=0.02) | 26.1% ❌ | 3.9% ✅ |
| **MIA Accuracy** | 50.2% ✅ | 67.4% ❌ |
| **MIA F1 Score** | 0.062 | 0.754 |
| **Memorization Gap** | +1.3% | +7.7% |
| **Training Epochs** | 10 (early stop) | 35 (full) |

> **Interpretation:** The adversarial model successfully defends against evasion by dropping the flip rate from 26.1% to 3.9%. However, it pays a steep privacy cost because it leaks membership information at nearly 70% accuracy, which is far above the 50% random-guessing baseline.

---

## 🏗️ Project Architecture

The codebase is organized into several key modules:
- `Victim_Model/`: Handles model training and provides a black-box API for evaluation.
- `fgsm_attack/`: Evaluates the evasion attack robustness.
- `Fluctuation_MIA/`: Implements our novel fluctuation-based membership inference attack.
- `Membership_Inference_Attack/`: Contains the standard confidence-based MIA for baseline comparison.
- `paper/`: Contains the research paper in IEEE LaTeX format, along with generated figures.

---

## ⚙️ Prerequisites

- **OS:** Linux (tested on Ubuntu 22.04)
- **Python:** 3.10 or higher
- **GPU:** NVIDIA GPU with CUDA 12.6+ (at least 8 GB VRAM recommended)
- **Disk Space:** ~50 GB for the full NIH dataset and extracted images
- **RAM:** 16 GB or more recommended

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KunalJindal19/Attacks-on-ML.git
cd Attacks-on-ML
```

### 2. Create Virtual Environment and Install Dependencies

You can run the setup script to handle everything automatically:
```bash
bash setup.sh
```

This will create a Python virtual environment called `mia_env`, upgrade pip, and install all dependencies from `requirements.txt`.

**Or manually:**
```bash
python3 -m venv mia_env
source mia_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** PyTorch is installed with CUDA 12.6 support. If you have a different CUDA version, please modify the `--extra-index-url` in `requirements.txt` accordingly.

---

## 🔄 Pipeline Execution

The full pipeline can be run automatically using:

```bash
source mia_env/bin/activate
bash run_pipeline.sh
```

Or you can run it step-by-step as described below.

---

### Step 1: Dataset Preparation

```bash
python Victim_Model/prepare_dataset.py
```

**What it does:**
1. Downloads all 12 zip files (~45 GB total) of the NIH Chest X-ray dataset from a HuggingFace mirror.
2. Extracts ~112,000 PNG chest X-ray images into `Victim_Model/images/`.
3. Builds `manifest.csv` containing image paths, labels, and splits.
4. Splits patients into **70% members** (training data) and **30% non-members** (unseen data).

> ⚠️ **Important:** The split is performed at the **patient level**, meaning all images from a given patient go into the same partition. This prevents data leakage and mirrors a realistic MIA threat model.

---

### Step 2: Training Victim Models

We train two DenseNet-121 victim models, both pre-trained on ImageNet and fine-tuned on the NIH Chest X-ray dataset, using different strategies.

#### 2a. Baseline Model (Standard Training)

```bash
python Victim_Model/train_baseline.py
```

**Training recipe:**
- **Architecture:** DenseNet-121
- **Optimizer:** AdamW with linear warmup and cosine annealing
- **Loss:** BCEWithLogitsLoss with per-class `pos_weight` to handle class imbalance
- **Regularization:** Dropout, weight decay, and early stopping
- **Mixed Precision:** AMP (FP16) on CUDA

#### 2b. Adversarial Model (FGSM Adversarial Training)

```bash
python Victim_Model/train_adversarial.py --epsilon 0.02 --no-augmentation --no-early-stopping
```

**What is different from baseline:**
The training loop generates FGSM adversarial examples on-the-fly and trains on a weighted combination of clean and adversarial loss. We intentionally disable early stopping to let the model overfit, demonstrating the robust overfitting phenomenon where the model memorizes adversarial patterns specific to its training data.

---

### Step 3: FGSM Evasion Attack

```bash
cd fgsm_attack
python attack_and_compare.py
```

**What it does:**
1. Loads both victim models via the black-box API.
2. Evaluates the FGSM attack at 5 different epsilon values.
3. Computes metrics like AUC, F1, accuracy, and flip rate.
4. Generates comparison reports.

You can also generate visual examples of the adversarial attacks:
```bash
python generate_adv_images.py
```

---

### Step 4: Membership Inference Attack

#### 4a. Fluctuation-Based MIA (Our Novel Approach)

```bash
cd Fluctuation_MIA
python run_fluctuation_attack.py
```

**What it does:**
1. Trains 8 adversarial shadow models on pseudo-labeled data from the victim API.
2. Extracts fluctuation features by measuring how much each disease prediction changes under adversarial perturbation.
3. Optionally adds a bootstrapped variance feature.
4. Trains 4 attack classifiers (Gradient Boosting, Random Forest, MLP, Logistic Regression).
5. Evaluates on a balanced set of members and non-members.

#### 4b. Standard Confidence-Based MIA (Baseline Comparison)

```bash
cd Membership_Inference_Attack
python run_attack.py
```

This runs the traditional shadow-model MIA using raw confidence scores instead of fluctuations, serving as a baseline to show that standard MIA cannot distinguish members from non-members on either model.

---

## 🧠 Attack Methodology

### FGSM Evasion Attack
The Fast Gradient Sign Method (Goodfellow et al., 2015) is a white-box attack that crafts adversarial examples by perturbing inputs in the direction of the gradient to maximize the loss. 

### Fluctuation-Based Membership Inference Attack
Our novel MIA exploits robust overfitting. The adversarial model has "seen" its training data during adversarial training and learned stable responses to perturbations of those specific images. For unseen images, perturbations cause larger output changes. This difference in fluctuation is the membership signal.

It works completely as a black-box attack since the attacker does not need to know how the victim was adversarially trained.

### Variance-Enhanced MIA
This enhancement computes the bootstrapped variance of the victim's max-confidence scores, capturing the overall confidence behavior to provide the attack classifier with richer context.

---

## 📊 Results

### Evasion Attack Results

| ε | Baseline AUC | Adversarial AUC | Baseline Flip Rate | Adversarial Flip Rate |
|:---:|:---:|:---:|:---:|:---:|
| 0.000 (clean) | 0.828 | 0.752 | — | — |
| 0.001 | 0.683 | 0.512 | 7.4% | 4.2% |
| 0.002 | 0.584 | 0.467 | 12.9% | 5.0% |
| 0.005 | 0.440 | 0.794 | 20.5% | 1.9% |
| 0.010 | 0.359 | 0.904 | 24.2% | 5.4% |
| 0.020 | 0.311 | 0.826 | 26.1% | 3.9% |

> The adversarial model maintains flip rates under 5.4% across all perturbation magnitudes, while the baseline suffers up to 26.1%.

### MIA Results — Baseline Victim

| Feature Strategy | Classifier | Accuracy | Precision | Recall | F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| Fluctuation | Gradient Boosting | 0.502 | 0.526 | 0.033 | 0.062 |
| Fluct + Variance | Gradient Boosting | 0.500 | 0.507 | 0.029 | 0.054 |

> All classifiers perform at or near random guessing (~50%), confirming the baseline model leaks no membership signal.

### MIA Results — Adversarial Victim (ε=0.02)

| Feature Strategy | Classifier | Accuracy | Precision | Recall | F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| Fluctuation | **Gradient Boosting** | **0.673** | 0.605 | 0.998 | **0.753** |
| Fluct + Variance | **Gradient Boosting** | **0.674** | 0.606 | 0.998 | **0.754** |

> **Best result:** Gradient Boosting with Fluctuation+Variance achieves **67.4% accuracy** and **0.754 F1**, confirming the adversarial model is vulnerable to membership inference.

---

## 📁 Repository Structure

- `Victim_Model/`: Contains scripts like `prepare_dataset.py`, `train_baseline.py`, and `api.py` for model training and serving.
- `fgsm_attack/`: Includes scripts like `attack_and_compare.py` for evasion attack evaluation.
- `Fluctuation_MIA/`: Houses `run_fluctuation_attack.py` and shadow model logic for the novel MIA.
- `Membership_Inference_Attack/`: Standard baseline MIA scripts.
- `paper/`: The LaTeX paper source code and generated figures.

---

## ⚙️ Configuration and Hyperparameters

### Victim Model Training
- **Architecture:** DenseNet-121
- **Optimizer:** AdamW
- **Learning rate:** 1e-4
- **Loss:** BCEWithLogitsLoss + pos_weight
- **Batch size:** 64
- **FGSM ε (training):** 0.02 (for adversarial model only)

### Shadow Model Training (MIA)
- **Number of shadow models:** 8
- **Architectures:** ResNet-18, MobileNetV3-Small, EfficientNet-B0
- **Training images per shadow:** 2,500
- **Epochs:** 40

### Attack Classifiers
- **Gradient Boosting:** n_estimators=200, max_depth=4, learning_rate=0.1
- **Random Forest:** n_estimators=200, max_depth=6

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{attacks-on-ml-2025,
  title={Robustness-Privacy Tradeoff in Adversarially Trained Medical Image Classifiers},
  author={Kunal Jindal},
  year={2025},
  howpublished={\url{https://github.com/KunalJindal19/Attacks-on-ML}}
}
```

---

## 🙏 Acknowledgements

- **NIH Chest X-ray Dataset:** Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", CVPR 2017
- **FGSM:** Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
- **Shadow Model MIA:** Shokri et al., "Membership Inference Attacks Against Machine Learning Models", IEEE S&P 2017
- **Robust Overfitting:** Rice et al., "Overfitting in Adversarially Robust Deep Learning", ICML 2020
- **Privacy Risks of Adversarial Training:** Song et al., "Privacy Risks of Securing Machine Learning Models Against Adversarial Examples", ACM CCS 2019

---

<div align="center">

**Built with ❤️ using PyTorch**

*For questions or issues, please open a GitHub issue.*

</div>