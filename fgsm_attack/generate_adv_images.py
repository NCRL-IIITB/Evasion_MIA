"""
generate_adv_images.py
======================
Generates and saves visual examples of FGSM adversarial attacks.
Picks configurable images from the dataset and generates their
adversarially perturbed versions across multiple epsilon values.

Usage
-----
  python fgsm_attack/generate_adv_images.py
  python fgsm_attack/generate_adv_images.py --num-images 2 --seed 42
  python fgsm_attack/generate_adv_images.py --image-paths path1.png path2.png
"""

import argparse
import ast
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VICTIM_DIR = os.path.join(PROJECT_ROOT, "Victim_Model")
MANIFEST_PATH = os.path.join(VICTIM_DIR, "manifest.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualizations")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

EPSILONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model building ──────────────────────────────────────────────────────────

def build_model(architecture: str, num_classes: int, use_dropout: bool = True) -> nn.Module:
    arch = architecture.lower()
    if arch == "densenet121":
        model = models.densenet121(weights=None)
        in_f = model.classifier.in_features
        if use_dropout:
            model.classifier = nn.Sequential(
                nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            model.classifier = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'")
    return model


def load_victim_model(model_path: str, meta_path: str) -> nn.Module:
    with open(meta_path, "r") as f:
        meta = json.load(f)

    architecture = meta.get("architecture", "densenet121")
    num_classes = int(meta.get("num_classes", 15))
    use_dropout = bool(meta.get("dropout", False))

    model = build_model(architecture, num_classes, use_dropout)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    return model


# ─── Helper functions ────────────────────────────────────────────────────────

def fgsm_attack(model, images, labels, epsilon, criterion):
    """Generate FGSM adversarial examples."""
    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    images.requires_grad = True

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign

    adv_images = torch.clamp(adv_images, -3, 3)
    return adv_images.detach()


def denormalize(tensor):
    """Reverses ImageNet normalization to get pixel values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def save_image(tensor, filepath):
    """Save a PyTorch tensor as an image file."""
    img = denormalize(tensor).squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C, H, W -> H, W, C
    img = (img * 255).astype(np.uint8)
    
    # Handle single channel (grayscale) vs 3 channel (RGB)
    if img.shape[2] == 1:
        img = img.squeeze(2)
        
    img_pil = Image.fromarray(img)
    img_pil.save(filepath)


# ─── Main Logic ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate FGSM adversarial images")
    parser.add_argument("--num-images", type=int, default=2, help="Number of random images to pick")
    parser.add_argument("--image-paths", nargs="+", help="Specific image paths to use (overrides --num-images)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--victim", type=str, default="baseline", choices=["baseline", "adversarial"],
                        help="Which victim model to attack")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving output images to: {OUTPUT_DIR}")

    # 1. Load model
    model_path = os.path.join(VICTIM_DIR, f"victim_{args.victim}.pth")
    meta_path = os.path.join(VICTIM_DIR, f"victim_{args.victim}_meta.json")
    print(f"Loading {args.victim} model...")
    model = load_victim_model(model_path, meta_path)
    criterion = nn.BCEWithLogitsLoss()

    # 2. Select images
    df = pd.read_csv(MANIFEST_PATH)
    
    if args.image_paths:
        selected_df = df[df["path"].isin(args.image_paths)]
        if len(selected_df) == 0:
            print("Provided image paths not found in manifest!")
            sys.exit(1)
    else:
        # Sample from non-members (test set)
        test_df = df[df["split"] == "nonmember"]
        selected_df = test_df.sample(n=args.num_images, random_state=args.seed)

    # 3. Transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # 4. Generate & Save
    for idx, row in selected_df.iterrows():
        path = row["path"]
        img_name = os.path.basename(path).split(".")[0]
        label_idx = ast.literal_eval(row["label_idx"])
        
        # Load and transform original image
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
            
        tensor_img = transform(img).unsqueeze(0).to(DEVICE)
        tensor_label = torch.tensor([label_idx], dtype=torch.float32).to(DEVICE)

        # Save original
        orig_path = os.path.join(OUTPUT_DIR, f"{img_name}_original.png")
        save_image(tensor_img, orig_path)
        print(f"\nProcessing {img_name}")
        print(f"  ✓ Saved {orig_path}")

        # Save adversarial for each epsilon
        for eps in EPSILONS:
            adv_tensor = fgsm_attack(model, tensor_img, tensor_label, eps, criterion)
            
            adv_path = os.path.join(OUTPUT_DIR, f"{img_name}_adv_{eps}.png")
            save_image(adv_tensor, adv_path)
            print(f"  ✓ Saved {adv_path}")

if __name__ == "__main__":
    main()
