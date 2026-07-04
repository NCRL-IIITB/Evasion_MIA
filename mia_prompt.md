# Complete Project Brief: Membership Inference Attack (MIA) on NIH Chest X-ray Dataset

## Overview

I need you to build a **Membership Inference Attack (MIA)** pipeline from scratch on the **NIH Chest X-ray dataset**. This project will demonstrate that an overfitted deep learning model leaks information about which data it was trained on. The pipeline should be clean, modular, and deployable on HuggingFace Spaces.

The project is a **re-implementation** of a working system I have previously built, but with a different folder structure and a different victim model architecture (I will specify the victim model below).

---

## What is a Membership Inference Attack (MIA)?

A Membership Inference Attack is a **privacy attack** on a machine learning model. Given a trained model and a data sample, the attacker asks: "Was this sample used to train the model?"

This attack exploits a specific weakness: **overfitting**. When a model overfits its training data, it outputs very high-confidence predictions for data points it was trained on (members) and lower-confidence predictions for data points it has never seen (non-members). By exploiting this confidence gap, an attacker can determine membership with above-random accuracy.

**Why is this important?** In medical imaging, a company might train a model on sensitive patient data (chest X-rays, MRIs, etc.). Even without access to the training data, an attacker can query the model's API and determine whether a specific patient's data was used to train the model. This is a serious privacy violation.

---

## Dataset: NIH Chest X-ray (Full 40GB Dataset)

- **Source:** NIH Chest X-ray dataset from HuggingFace: `https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset`
- **Total Images:** ~112,120 PNG chest X-ray images (the full 40GB dataset, split across 12 zip files: `images_001.zip` to `images_012.zip`)
- **Label File:** `Data_Entry_2017.csv` (I will provide this file; it is ~8.6MB and maps each image filename to its disease labels)
- **Task:** Multi-label classification across **15 classes** (14 diseases + "No Finding")

**The 15 disease classes are (in this exact order):**
```
Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema,
Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening,
Cardiomegaly, Nodule, Mass, Hernia, No Finding
```

**Label format in the CSV:** Each row has an "Image Index" column (filename like `00000001_000.png`) and a "Finding Labels" column (pipe-separated diseases like `Cardiomegaly|Effusion` or `No Finding`). You must parse this into a 15-dimensional multi-hot binary vector for training.

---

## CRITICAL: The 50/50 Data Split Rule (Read This Very Carefully)

This is the most important design constraint of the entire project. **Do NOT deviate from this logic under any circumstances.**

### The Scenario Being Simulated:
In the real world, when an attacker executes a MIA, they have access to the **full dataset** (e.g., a public dataset like NIH), but they do NOT know which 50% of that data the victim model was actually trained on.

### The Split Rule:
1. Take the **complete full dataset** (all ~112,000 images from the full 40GB dataset).
2. **Randomly sample exactly 50%** of images from the entire dataset. These are the **MEMBERS** — the data the victim model will be trained on.
3. The remaining **50%** are **NON-MEMBERS** — data the victim model has NEVER seen.
4. Train the victim model on ONLY the member images.
5. After the victim is trained, **shuffle all images** (members + non-members) together into a single pool. The attacker now has access to all images but does NOT know which ones were members.
6. The shadow models are trained on subsets of this full shuffled pool (both members and non-members are visible to the attacker, but the attacker doesn't know which is which).

### WHAT YOU MUST NOT DO:
- **DO NOT** take a small subset of the data first and then split that subset into members/non-members. For example, do NOT do: "take first 2 zip files → split those 15k images 60/40". This is wrong because it does not reflect the real-world scenario.
- **DO NOT** use only a subset of the full 40GB dataset unless instructed explicitly by the user.
- The 50% sampling must be from the **ENTIRE** `Data_Entry_2017.csv` file (all ~112,120 rows that have corresponding images).

### Implementation Logic:
```python
# CORRECT APPROACH:
# 1. Load all image paths from Data_Entry_2017.csv that have a matching downloaded image
# 2. Shuffle them with a fixed random seed (e.g., 42)
# 3. Take first 50% as MEMBER_PATHS, second 50% as NONMEMBER_PATHS
# 4. Train victim ONLY on MEMBER_PATHS
# 5. For the attack phase, the attacker's pool = all images shuffled (no labels of member/non-member)
# 6. For EVALUATION: use held-out subset of known members and known non-members
```

---

## Pipeline Architecture (4 Scripts)

The pipeline consists of 4 main scripts that run sequentially:

### Script 1: `prepare_dataset.py`
- Downloads all 12 ZIP files from HuggingFace (`images_001.zip` to `images_012.zip`)
- Extracts all images into a common `images/` folder
- Reads `Data_Entry_2017.csv` (provided locally, do NOT download)
- Performs the 50/50 member/non-member split on the FULL dataset
- Parses multi-hot labels for each image (15-dimensional vector)
- Writes a manifest CSV: `manifest.csv` with columns: `path, label_vec, split` where:
  - `path` = absolute path to the image file
  - `label_vec` = the 15-dimensional multi-hot label as a string or list
  - `split` = "member" or "nonmember"

### Script 2: `train_victim.py`
- Loads only the MEMBER rows from `manifest.csv`
- Splits members 85%/15% into training/validation sets (validation is for monitoring only)
- Trains the victim model with **intentional overfitting**: NO dropout, NO weight decay, NO data augmentation, full fine-tuning
- Loss function: `BCEWithLogitsLoss` (multi-label)
- Optimizer: Adam
- Saves: `victim.pth` (model weights) + `victim_meta.json` (num_classes, class_names, img_size)
- Logs epoch-by-epoch: train_loss, train_acc, val_loss, val_acc, and the "memorization gap" (train_acc - val_acc)

### Script 3: `api.py`
- A wrapper class `VictimAPI` that loads `victim.pth` and exposes a single method:
  - `predict(image_paths: np.ndarray) -> np.ndarray` — takes an array of file paths, returns `(N, 15)` sigmoid confidence scores
- This simulates the "black-box API" that an attacker would query

### Script 4: `run_attack.py`
- Implements two MIA variants:
  1. **Baseline Shadow Model MIA**
  2. **Variance-Enhanced Shadow Model MIA**
- Full details in the Shadow Model section below

---

## Victim Model Architecture

> **IMPORTANT:** I want to experiment with a different victim model from my previous run. You should ask me which architecture I want (or I will specify it in a follow-up). However, the training procedure MUST follow these rules regardless of the architecture:
> - Pretrained on ImageNet (use pretrained=True)
> - Replace the final classification head with a custom multi-label head outputting 15 logits
> - NO dropout, NO weight decay
> - Loss: `BCEWithLogitsLoss`
> - Optimizer: Adam, lr=1e-4
> - Full fine-tune (all layers trainable, not just the head)

**In my previous experiment (for reference), I used DenseNet-121 with this head:**
```python
model.classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 15)
)
```
**This achieved:** train_acc=99.99%, val_acc=93.61%, memorization gap=+6.39%, and led to attack accuracy of 73.25% (Variance MIA).

---

## Shadow Model Architecture (How the Attack Works)

The attack follows the **Shokri et al. Shadow Model Attack** paradigm. Here is exactly how it works:

### Step 1: Train Shadow Models on Images
Shadow models are CNN classifiers trained exactly like the victim — on **raw images**, with the same task (15-class multi-label). They are trained on subsets sampled from the full shuffled pool (attacker's view). Their purpose is to mimic the victim model's behavior so we can collect training data for the attack model.

**Shadow model architectures used (cycling through):**
- ResNet-18
- MobileNetV3-Small
- EfficientNet-B0

**Shadow model training parameters (from my successful run):**
- Number of shadow models: 8
- Training images per shadow model: 2,500 (sampled from pool)
- Epochs per shadow model: 15
- Batch size: 32
- Learning rate: 1e-3
- Loss: `BCEWithLogitsLoss`
- Labels for shadow training: Derived by querying the victim API — the victim's sigmoid scores are thresholded at 0.5 to produce pseudo multi-hot labels

**Critical detail on shadow model training:**
```python
# For each shadow model:
# 1. Sample 2500 random images from the pool (the attacker's shuffled full dataset)
# 2. Query the VICTIM API on those 2500 images to get pseudo-labels
pseudo_labels = (victim_api.predict(train_paths) > 0.5).astype(np.float32)
# 3. Train the shadow CNN on those 2500 images with those pseudo-labels
shadow_model.fit(train_paths, pseudo_labels)
```

### Step 2: Build the Attack Dataset
After training each shadow model, we query it on:
- Images that WERE in its training set → label them as "member" (1)
- Images that were NOT in its training set → label them as "non-member" (0)

For each image queried, we record the shadow model's **15 sigmoid confidence scores** as the feature vector. This gives us a dataset with:
- Features: 15-dimensional confidence score vectors
- Labels: binary (1=member, 0=non-member)

**From 8 shadow models × 2500 training images → ~20,000 rows in the attack training dataset** (10,000 member, 10,000 non-member)

### Step 3: Train the Attack Model
Train a `GradientBoostingClassifier` (sklearn) on the attack dataset:
- `n_estimators=100, learning_rate=0.1`
- Input: 15-dimensional confidence vector from victim API
- Output: Binary prediction (member=1, non-member=0)

### Step 4: Evaluate the Attack
Run the attack on a held-out evaluation set of KNOWN members and KNOWN non-members. Query the victim API on each evaluation image to get its 15-dimensional confidence vector, then feed it to the attack model to predict membership.

**Evaluation metrics:** Accuracy, Precision, Recall, F1.

---

## Attack 2: Variance-Enhanced MIA

This is an improvement on the baseline that adds one extra feature: the **variance of the maximum confidence score across all shadow models**.

**Logic:** For each evaluation image:
1. Query ALL shadow models on it
2. For each shadow model, record the MAX of its 15 sigmoid scores (its "most confident" class score)
3. Compute the VARIANCE of these max scores across all shadow models
4. Use 16 features: 15 sigmoid scores from the VICTIM model + 1 variance_of_max score

**Why this works:** Members of the victim model are typically the same types of images the shadow models are confident about (since the shadow models mimic the victim). The variance across shadow models for members tends to be lower and more consistent, while for non-members it's higher and more scattered.

**The attack dataset for Variance MIA:**
```
Features: [class_0, class_1, ..., class_14, variance_of_max] (16 features)
Labels: binary (1=member, 0=non-member)
```

---

## Benchmark Results (From My Previous Experiments)

I have run two experiments. Use these as reference benchmarks:

### Experiment 1: Small Dataset Run (15,000 images, 2 zip files, 60/40 split)
*(Note: this used the WRONG split method — it only used 2 zip files and used 60/40 instead of 50/50 from the full dataset. Results are here for reference only.)*
- Victim: DenseNet-121, 30 epochs, train_acc=99.99%, val_acc=93.61%, gap=+6.39%
- Pool: 4,000 images (2,000 member + 2,000 non-member)
- Shadow models: 6, 1,500 images each, 15 epochs
- Attack model training data: 9,000 rows (4,500 member, 4,500 non-member)
- **Baseline MIA Accuracy: 71.50%**, Precision: 0.6369, Recall: 1.000, F1: 0.778
- **Variance MIA Accuracy: 73.25%**, Precision: 0.6515, Recall: 1.000, F1: 0.789
- Total runtime: ~61 minutes on NVIDIA A10G

### Experiment 2: Large Dataset Run (15,000 images, 2 zip files, 60/40 split, larger attack)
*(Same wrong split method but larger attack configuration)*
- Victim: DenseNet-121, 30 epochs, train_acc=99.94%, val_acc=93.39%, gap=+6.56%
- Pool: 6,000 images (3,000 member + 3,000 non-member)
- Shadow models: 8, 2,500 images each, 15 epochs
- **Similar attack accuracy expected (results file was truncated)**
- Confidence gap (member vs non-member): +0.1685

**Key diagnostic from Experiment 1:**
- Mean max-confidence on members: 0.9999
- Mean max-confidence on non-members: 0.8303
- Confidence gap: +0.1697 (this is the signal the attack model exploits)

---

## Important Implementation Notes

### 1. Data Loading from ShadowDataset
The `ShadowDataset` PyTorch class must load images from **file paths**, NOT from arrays. Each shadow model's `fit()` method receives an array of file path strings and a 2D array of multi-hot labels. It internally creates a Dataset that opens images with PIL, converts to RGB, and applies the transform pipeline:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

### 2. VictimAPI must also take file paths
The victim model's `predict()` method takes a numpy array of file path strings and returns `(N, 15)` sigmoid scores. It processes them in batches:
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    # X is an array of file path strings
    # Returns (N, 15) sigmoid confidence scores
```

### 3. Stdout flushing for HuggingFace
The code runs on HuggingFace Spaces (like a long-running subprocess). Print statements must flush stdout regularly so the log stream doesn't hang the connection:
```python
import sys
print("...", flush=True)
sys.stdout.flush()
```

### 4. Avoid binary files in git
The `Data_Entry_2017.csv` file is OK (it's a text CSV). But never commit `.xlsx`, `.pth` model weights, or large zip files to the HuggingFace Space. The model weights are saved during the run and do not need to be in the repo.

### 5. HuggingFace Space entrypoint
The `app.py` file (Gradio app) should sequentially call:
1. `prepare_dataset.main()` — downloads data, creates manifest
2. `train_victim.main()` — trains the victim
3. `run_attack.main()` — runs both MIA attacks and prints results

The Gradio interface should just trigger this pipeline and stream the output to the user.

### 6. Keep the manifest as source of truth
`manifest.csv` is the bridge between all scripts. It must contain:
- `path`: absolute path to the image on disk
- `split`: "member" or "nonmember"
- `label_vec`: the 15-element multi-hot vector (stored as a Python list string, e.g., `"[1,0,0,1,0,0,0,1,0,0,0,0,0,0,0]"`)
- `label`: the human-readable label string from the CSV (e.g., "Atelectasis|Effusion")

---

## Recommended Parameters for the Next Run (Full 40GB Dataset)

Since you will be running on the full dataset (~56,000 members, ~56,000 non-members), here are the recommended parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Victim epochs | 30 | With 56k images per epoch, 30 epochs is sufficient to overfit |
| Victim batch size | 64 | Standard |
| Victim learning rate | 1e-4 | Adam |
| Shadow model count | 8 | Diversity without excessive cost |
| Shadow model dataset size | 2,500 | Per shadow model |
| Shadow model epochs | 15 | Per shadow model |
| Shadow model batch size | 32 | Standard |
| Pool size (for attack training) | 6,000 (3,000 member + 3,000 non-member) | Sufficient for shadow training |
| Eval size | 4,000 (2,000 member + 2,000 non-member) | Statistically significant evaluation |
| Attack model | GradientBoostingClassifier | n_estimators=100, lr=0.1 |
| Attack training data size | ~20,000 rows | 8 models × ~2,500 samples |

---

## Expected Final Output

After running the complete pipeline, the output should look like:
```
  Attack Method                              Acc     Prec      Rec       F1
  -----------------------------------------------------------------------
  Baseline Shadow Model (Gradient Boosting)  0.71+   0.63+    1.0000   0.77+
  Variance Shadow Model (Gradient Boosting)  0.73+   0.65+    1.0000   0.79+

  Random baseline: 0.5000
```

The target is **>70% attack accuracy** (well above the 50% random baseline). This confirms the model has memorized its training data and is vulnerable to MIA.
