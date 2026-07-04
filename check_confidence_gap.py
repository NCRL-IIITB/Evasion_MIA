import os
import sys
import numpy as np
import pandas as pd

# Make sure we can import VictimAPI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Victim_Model.api import VictimAPI

MODEL_PATH = "Victim_Model/victim_adversarial_eps002_noaug.pth"
MANIFEST_PATH = "Victim_Model/manifest.csv"

def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Split into members and non-members
    members = df[df["split"] == "member"]["path"].values
    nonmembers = df[df["split"] == "nonmember"]["path"].values
    
    # Sample 1000 of each
    np.random.seed(42)
    members_sample = np.random.choice(members, 1000, replace=False)
    nonmembers_sample = np.random.choice(nonmembers, 1000, replace=False)
    
    print(f"Loading victim model: {MODEL_PATH}...")
    api = VictimAPI(MODEL_PATH, batch_size=64)
    
    print("Running inference on 1000 Members...")
    member_scores = api.predict(members_sample)
    
    print("Running inference on 1000 Non-Members...")
    nonmember_scores = api.predict(nonmembers_sample)
    
    member_max = np.max(member_scores, axis=1).mean()
    member_min = np.min(member_scores, axis=1).mean()
    
    nonmember_max = np.max(nonmember_scores, axis=1).mean()
    nonmember_min = np.min(nonmember_scores, axis=1).mean()
    
    print("\n" + "="*40)
    print("           CONFIDENCE GAP RESULTS")
    print("="*40)
    print(f"Members (n=1000):")
    print(f"  Average Max Confidence : {member_max:.4f}")
    print(f"  Average Min Confidence : {member_min:.4f}")
    print()
    print(f"Non-Members (n=1000):")
    print(f"  Average Max Confidence : {nonmember_max:.4f}")
    print(f"  Average Min Confidence : {nonmember_min:.4f}")
    print("="*40)
    
    # Let's also print the difference
    diff_max = member_max - nonmember_max
    diff_min = member_min - nonmember_min
    print(f"Max Confidence Gap: {diff_max:+.4f}")
    print(f"Min Confidence Gap: {diff_min:+.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
