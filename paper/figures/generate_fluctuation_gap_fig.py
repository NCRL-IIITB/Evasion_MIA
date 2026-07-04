import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # Data from Table V
    classes = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural Thick.', 'Cardiomegaly', 'Nodule', 'Mass',
        'Hernia', 'No Finding'
    ]
    
    members = [
        0.1992, 0.1180, 0.2050, 0.0268, 0.0166, 0.0137, 0.0161, 0.0868, 
        0.0102, 0.0203, 0.0196, 0.0716, 0.0396, 0.0008, 0.3800
    ]
    
    non_members = [
        0.2058, 0.1025, 0.1970, 0.0279, 0.0175, 0.0131, 0.0202, 0.0848, 
        0.0120, 0.0201, 0.0195, 0.0782, 0.0410, 0.0007, 0.3814
    ]
    
    avg_member = 0.0816
    avg_non_member = 0.0814
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot bars
    rects1 = ax.bar(x - width/2, members, width, label='Members', color='#3498db', edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, non_members, width, label='Non-Members', color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    # Customize axes
    ax.set_ylabel('Average Fluctuation $|f(x)_c - f(x_{adv})_c|$', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title(f'Per-Class Fluctuation Gap Under Adversarial Training ($\epsilon = 0.01$)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add horizontal lines for averages
    ax.axhline(y=avg_member, color='#2980b9', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(y=avg_non_member, color='#c0392b', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add annotations for the averages
    ax.text(len(classes) - 0.5, avg_member + 0.005, f'Avg Members: {avg_member}', color='#2980b9', fontsize=12, fontweight='bold', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    ax.text(len(classes) - 0.5, avg_non_member + 0.005, f'Avg Non-Members: {avg_non_member}', color='#c0392b', fontsize=12, fontweight='bold', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Add legend
    ax.legend(fontsize=14, loc='upper left')
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Formatting
    plt.tight_layout()
    
    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fluctuation_gap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
