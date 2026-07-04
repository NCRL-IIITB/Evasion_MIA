import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VICTIM_DIR = os.path.join(BASE_DIR, 'Victim_Model')
sys.path.insert(0, VICTIM_DIR)
from api import VictimAPI

MANIFEST_PATH = os.path.join(VICTIM_DIR, 'manifest.csv')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
           'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
           'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']

EPSILON = 0.01
NUM_SAMPLES = 2000  # Evaluate on 2000 members and 2000 non-members for speed

class FastDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, 'path']
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Get labels
        import ast
        label_idx = ast.literal_eval(self.df.loc[idx, 'label_idx'])
        target = torch.zeros(15)
        target[label_idx] = 1.0
        
        return img, target

def evaluate_fluctuation(model_path, members_df, non_members_df, device):
    print(f"\nEvaluating: {os.path.basename(model_path)}")
    api = VictimAPI(model_path, num_classes=15, batch_size=64)
    model = api.model
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def process_subset(df, name):
        dataset = FastDataset(df, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        total_fluctuation = torch.zeros(15).to(device)
        total_clean_conf = torch.zeros(15).to(device)
        total_samples = 0
        
        print(f"  Processing {name} ({len(df)} samples)...")
        start_t = time.time()
        
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            images.requires_grad = True
            
            # Forward clean
            logits_clean = model(images)
            probs_clean = torch.sigmoid(logits_clean)
            
            # Generate adversarial
            loss = loss_fn(logits_clean, targets)
            model.zero_grad()
            loss.backward()
            
            grad_sign = images.grad.data.sign()
            images_adv = torch.clamp(images + EPSILON * grad_sign, -3.0, 3.0).detach()
            
            # Forward adversarial
            with torch.no_grad():
                logits_adv = model(images_adv)
                probs_adv = torch.sigmoid(logits_adv)
            
            # Accumulate
            fluctuation = torch.abs(probs_clean.detach() - probs_adv).sum(dim=0)
            total_fluctuation += fluctuation
            total_clean_conf += probs_clean.detach().sum(dim=0)
            total_samples += images.size(0)
            
        print(f"    Done in {time.time()-start_t:.1f}s")
        return (total_fluctuation / total_samples).cpu().numpy(), (total_clean_conf / total_samples).cpu().numpy()

    fl_mem, conf_mem = process_subset(members_df, "Members")
    fl_nmem, conf_nmem = process_subset(non_members_df, "Non-Members")
    
    return fl_mem, fl_nmem, conf_mem, conf_nmem

def main():
    df = pd.read_csv(MANIFEST_PATH)
    
    members = df[df['split'] == 'member'].sample(n=NUM_SAMPLES, random_state=42)
    non_members = df[df['split'] == 'nonmember'].sample(n=NUM_SAMPLES, random_state=42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models = {
        "Adversarial": os.path.join(VICTIM_DIR, 'victim_adversarial_eps01_noaug.pth'),
        "Baseline": os.path.join(VICTIM_DIR, 'victim_baseline.pth')
    }
    
    results = {}
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Model missing: {path}")
            continue
            
        fl_mem, fl_nmem, conf_mem, conf_nmem = evaluate_fluctuation(path, members, non_members, device)
        results[name] = {
            'fl_mem': fl_mem, 'fl_nmem': fl_nmem,
            'conf_mem': conf_mem, 'conf_nmem': conf_nmem
        }
        
    print("\n" + "="*60)
    print("RESULTS (LaTeX Table Format)")
    print("="*60)
    
    for name, res in results.items():
        print(f"\n--- {name} Model ---")
        fl_mem = res['fl_mem']
        fl_nmem = res['fl_nmem']
        
        print(f"{'Class':<20} & {'Members':<10} & {'Non-Members':<12} & {'Gap (M-NM)':<10} \\\\")
        print("-" * 60)
        for i, cls in enumerate(CLASSES):
            gap = fl_mem[i] - fl_nmem[i]
            sign = "+" if gap > 0 else "-"
            print(f"{cls:<20} & {fl_mem[i]:.4f}     & {fl_nmem[i]:.4f}       & {sign}{abs(gap):.4f}     \\\\")
            
        print("-" * 60)
        avg_mem = fl_mem.mean()
        avg_nmem = fl_nmem.mean()
        avg_gap = avg_mem - avg_nmem
        sign = "+" if avg_gap > 0 else "-"
        print(f"{'Average':<20} & {avg_mem:.4f}     & {avg_nmem:.4f}       & {sign}{abs(avg_gap):.4f}     \\\\")

if __name__ == "__main__":
    main()
