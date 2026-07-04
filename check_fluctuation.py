import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Victim_Model.api import VictimAPI

MODEL_PATH = "Victim_Model/victim_overfit.pth"
MANIFEST_PATH = "Victim_Model/manifest.csv"
EPSILON = 0.01

DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema",   "Fibrosis",       "Effusion",     "Pneumonia",    "Pleural_Thickening",
    "Cardiomegaly","Nodule",          "Mass",         "Hernia",       "No Finding",
]

class NIHInferenceDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

def compute_fluctuation(api, paths, labels):
    device = api.device
    model = api.model
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = NIHInferenceDataset(paths, labels, api.transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    all_clean_probs = []
    all_adv_probs = []
    
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        # Clean predictions
        model.eval()
        with torch.no_grad():
            clean_logits = model(images)
            clean_probs = torch.sigmoid(clean_logits).cpu().numpy()
            all_clean_probs.append(clean_probs)
            
        # Generate Adv examples
        model.train() # Need gradients
        images_grad = images.clone().detach().requires_grad_(True)
        logits = model(images_grad)
        loss = criterion(logits, targets)
        loss.backward()
        
        grad_sign = images_grad.grad.data.sign()
        adv_images = torch.clamp(images + EPSILON * grad_sign, -3.0, 3.0).detach()
        
        # Adv predictions
        model.eval()
        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_probs = torch.sigmoid(adv_logits).cpu().numpy()
            all_adv_probs.append(adv_probs)
            
    clean_probs = np.vstack(all_clean_probs)
    adv_probs = np.vstack(all_adv_probs)
    
    # Calculate absolute difference
    abs_diff = np.abs(adv_probs - clean_probs)
    
    # Average across all datapoints for each class
    mean_abs_diff_per_class = np.mean(abs_diff, axis=0)
    
    return mean_abs_diff_per_class

def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH)
    
    df_members = df[df["split"] == "member"].reset_index(drop=True)
    df_nonmembers = df[df["split"] == "nonmember"].reset_index(drop=True)
    
    np.random.seed(42)
    sample_m = df_members.sample(n=1000, random_state=42).reset_index(drop=True)
    sample_nm = df_nonmembers.sample(n=1000, random_state=42).reset_index(drop=True)
    
    def get_data(sampled_df):
        paths = sampled_df["path"].values
        labels = np.array([ast.literal_eval(l) for l in sampled_df["label_idx"]], dtype=float)
        return paths, labels
        
    paths_m, labels_m = get_data(sample_m)
    paths_nm, labels_nm = get_data(sample_nm)
    
    print(f"Loading victim model: {MODEL_PATH}...")
    api = VictimAPI(MODEL_PATH, batch_size=64)
    
    print("Computing fluctuation for 1000 Members (eps=0.01)...")
    fluc_m = compute_fluctuation(api, paths_m, labels_m)
    
    print("Computing fluctuation for 1000 Non-Members (eps=0.01)...")
    fluc_nm = compute_fluctuation(api, paths_nm, labels_nm)
    
    print("\n" + "="*70)
    print("       MODEL FLUCTUATION RESULTS (Avg Abs Diff per Class)")
    print("="*70)
    print(f"{'Class Name':<20} | {'Members (1000)':<15} | {'Non-Members (1000)':<15} | {'Gap (M - NM)'}")
    print("-" * 70)
    
    for i, cls in enumerate(DISEASE_CLASSES):
        print(f"{cls:<20} | {fluc_m[i]:<15.6f} | {fluc_nm[i]:<15.6f} | {fluc_m[i] - fluc_nm[i]:+.6f}")
        
    print("-" * 70)
    print(f"{'AVERAGE ACROSS CLASSES':<20} | {np.mean(fluc_m):<15.6f} | {np.mean(fluc_nm):<15.6f} | {np.mean(fluc_m) - np.mean(fluc_nm):+.6f}")
    print("="*70)

if __name__ == "__main__":
    main()
