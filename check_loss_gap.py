import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score

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

def compute_losses(api, paths, labels):
    device = api.device
    model = api.model
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    dataset = NIHInferenceDataset(paths, labels, api.transform)
    loader = DataLoader(dataset, batch_size=api.batch_size, shuffle=False, num_workers=0)
    
    all_clean_losses = []
    all_adv_losses = []
    
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        # Clean predictions
        model.eval()
        with torch.no_grad():
            clean_logits = model(images)
            clean_loss = criterion(clean_logits, targets).mean(dim=1).cpu().numpy()
            all_clean_losses.append(clean_loss)
            
        # Generate Adv examples
        model.train() # Need gradients
        images_grad = images.clone().detach().requires_grad_(True)
        logits = model(images_grad)
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        loss.backward()
        
        grad_sign = images_grad.grad.data.sign()
        adv_images = torch.clamp(images + EPSILON * grad_sign, -3.0, 3.0).detach()
        
        # Adv predictions
        model.eval()
        with torch.no_grad():
            adv_logits = model(adv_images)
            adv_loss = criterion(adv_logits, targets).mean(dim=1).cpu().numpy()
            all_adv_losses.append(adv_loss)
            
    return np.concatenate(all_clean_losses), np.concatenate(all_adv_losses)

def find_best_threshold(losses_m, losses_nm):
    # MIA predicts Member if Loss < threshold
    y_true = np.concatenate([np.ones_len(losses_m), np.zeros_len(losses_nm)])
    y_scores = np.concatenate([-losses_m, -losses_nm]) # negative loss so higher score = member
    
    auc = roc_auc_score(y_true, y_scores)
    
    best_acc = 0
    best_thresh = 0
    
    # Try 100 thresholds between min and max loss
    thresholds = np.linspace(min(min(losses_m), min(losses_nm)), max(max(losses_m), max(losses_nm)), 1000)
    for t in thresholds:
        preds = np.concatenate([losses_m < t, losses_nm < t])
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    return auc, best_acc, best_thresh

def np_ones_len(arr): return np.ones(len(arr))
def np_zeros_len(arr): return np.zeros(len(arr))
np.ones_len = np_ones_len
np.zeros_len = np_zeros_len

def main():
    df = pd.read_csv(MANIFEST_PATH)
    
    df_members = df[df["split"] == "member"].reset_index(drop=True)
    df_nonmembers = df[df["split"] == "nonmember"].reset_index(drop=True)
    
    # Let's test on 2000 samples to be fast but representative
    np.random.seed(42)
    sample_m = df_members.sample(n=2000, random_state=42).reset_index(drop=True)
    sample_nm = df_nonmembers.sample(n=2000, random_state=42).reset_index(drop=True)
    
    def get_data(sampled_df):
        paths = sampled_df["path"].values
        labels = np.array([ast.literal_eval(l) for l in sampled_df["label_idx"]], dtype=float)
        return paths, labels
        
    paths_m, labels_m = get_data(sample_m)
    paths_nm, labels_nm = get_data(sample_nm)
    
    api = VictimAPI(MODEL_PATH, batch_size=64)
    
    clean_m, adv_m = compute_losses(api, paths_m, labels_m)
    clean_nm, adv_nm = compute_losses(api, paths_nm, labels_nm)
    
    print("\n" + "="*50)
    print("      MIA THRESHOLD ATTACK RESULTS (LOSS)")
    print("="*50)
    
    auc_clean, acc_clean, t_clean = find_best_threshold(clean_m, clean_nm)
    print("CLEAN LOSS ATTACK:")
    print(f"  AUC      : {auc_clean:.4f}")
    print(f"  Best Acc : {acc_clean:.4f} (at threshold {t_clean:.4f})")
    
    auc_adv, acc_adv, t_adv = find_best_threshold(adv_m, adv_nm)
    print("\nADVERSARIAL LOSS ATTACK:")
    print(f"  AUC      : {auc_adv:.4f}")
    print(f"  Best Acc : {acc_adv:.4f} (at threshold {t_adv:.4f})")
    print("="*50)

if __name__ == "__main__":
    main()
