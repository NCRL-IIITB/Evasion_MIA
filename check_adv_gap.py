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

MODEL_PATH = "Victim_Model/victim_baseline.pth"
MANIFEST_PATH = "Victim_Model/manifest.csv"
EPSILON = 0.02

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

def compute_adv_metrics(api, paths, labels):
    device = api.device
    model = api.model
    criterion = nn.BCEWithLogitsLoss()
    
    dataset = NIHInferenceDataset(paths, labels, api.transform)
    loader = DataLoader(dataset, batch_size=api.batch_size, shuffle=False, num_workers=0)
    
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
            
    return np.vstack(all_clean_probs), np.vstack(all_adv_probs)


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
    
    print("Running adversarial inference on Members...")
    clean_m, adv_m = compute_adv_metrics(api, paths_m, labels_m)
    
    print("Running adversarial inference on Non-Members...")
    clean_nm, adv_nm = compute_adv_metrics(api, paths_nm, labels_nm)
    
    # Calculate accuracy
    acc_clean_m = np.mean((clean_m > 0.5) == labels_m)
    acc_adv_m = np.mean((adv_m > 0.5) == labels_m)
    
    acc_clean_nm = np.mean((clean_nm > 0.5) == labels_nm)
    acc_adv_nm = np.mean((adv_nm > 0.5) == labels_nm)
    
    print("\n" + "="*55)
    print("           ROBUST OVERFITTING RESULTS")
    print("="*55)
    print(f"Members (n=1000):")
    print(f"  Clean Accuracy : {acc_clean_m * 100:.4f}%")
    print(f"  Adv Accuracy   : {acc_adv_m * 100:.4f}%")
    print(f"Non-Members (n=1000):")
    print(f"  Clean Accuracy : {acc_clean_nm * 100:.4f}%")
    print(f"  Adv Accuracy   : {acc_adv_nm * 100:.4f}%")
    print("="*55)
    print(f"Clean Gap : {(acc_clean_m - acc_clean_nm)*100:+.4f}%")
    print(f"Adv Gap   : {(acc_adv_m - acc_adv_nm)*100:+.4f}%")
    print("="*55)

if __name__ == "__main__":
    main()
