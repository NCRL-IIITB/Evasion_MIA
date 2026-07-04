import os
import sys
import ast
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Victim_Model.api import VictimAPI

MODEL_PATH = "Victim_Model/victim_adversarial_eps002_noaug.pth"
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
        labels = np.array([ast.literal_eval(l) for l in sampled_df["label_idx"]], dtype=float)
        return paths, labels
        
    paths_m, labels_m = get_labels(sample_m)
    paths_nm, labels_nm = get_labels(sample_nm)
    
    print(f"Loading victim model: {MODEL_PATH}...")
    api = VictimAPI(MODEL_PATH, batch_size=64)
    
    print("Running inference on 1000 Members...")
    scores_m = api.predict(paths_m)
    
    print("Running inference on 1000 Non-Members...")
    scores_nm = api.predict(paths_nm)
    
    # Calculate accuracy
    preds_m = (scores_m > 0.5).astype(float)
    acc_m = np.mean(preds_m == labels_m)
    
    preds_nm = (scores_nm > 0.5).astype(float)
    acc_nm = np.mean(preds_nm == labels_nm)
    
    print("\n" + "="*50)
    print("           ACCURACY GAP RESULTS")
    print("="*50)
    print(f"Members (n=1000) Accuracy     : {acc_m * 100:.4f}%")
    print(f"Non-Members (n=1000) Accuracy : {acc_nm * 100:.4f}%")
    print(f"Accuracy Gap (Memorization)   : {(acc_m - acc_nm) * 100:+.4f}%")
    print("="*50)

if __name__ == "__main__":
    main()
