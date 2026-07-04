import os
import sys
import ast
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Victim_Model.api import VictimAPI

MODEL_PATH = "Victim_Model/victim_adversarial_eps01_noaug.pth"
MANIFEST_PATH = "Victim_Model/manifest.csv"

def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Filter by split
    df_members = df[df["split"] == "member"].reset_index(drop=True)
    df_nonmembers = df[df["split"] == "nonmember"].reset_index(drop=True)
    
    # Sample 1000
    sample_m = df_members.sample(n=1000, random_state=42).reset_index(drop=True)
    sample_nm = df_nonmembers.sample(n=1000, random_state=42).reset_index(drop=True)
    
    # Extract paths and true labels
    def get_labels(sampled_df):
        paths = sampled_df["path"].values
        labels = np.array([ast.literal_eval(l) for l in sampled_df["label_idx"]], dtype=bool)
        return paths, labels
        
    paths_m, labels_m = get_labels(sample_m)
    paths_nm, labels_nm = get_labels(sample_nm)
    
    print(f"Loading victim model: {MODEL_PATH}...")
    api = VictimAPI(MODEL_PATH, batch_size=64)
    class_names = api.label_names
    if not class_names:
        class_names = [f"Class_{i}" for i in range(15)]
        
    print("Running inference on 1000 Members...")
    scores_m = api.predict(paths_m)
    
    print("Running inference on 1000 Non-Members...")
    scores_nm = api.predict(paths_nm)
    
    print("\n" + "="*85)
    print(f"{'Class Name':<20} | {'Members (Avg Score when True)':<30} | {'Non-Members (Avg Score when True)'}")
    print("-" * 85)
    
    for c in range(15):
        # Members
        mask_m = labels_m[:, c]
        count_m = np.sum(mask_m)
        avg_m = np.mean(scores_m[mask_m, c]) if count_m > 0 else 0.0
            
        # Non-members
        mask_nm = labels_nm[:, c]
        count_nm = np.sum(mask_nm)
        avg_nm = np.mean(scores_nm[mask_nm, c]) if count_nm > 0 else 0.0
            
        print(f"{class_names[c]:<20} | {avg_m:.4f} (n={count_m:<4})                   | {avg_nm:.4f} (n={count_nm:<4})")
        
    print("="*85)

if __name__ == "__main__":
    main()
