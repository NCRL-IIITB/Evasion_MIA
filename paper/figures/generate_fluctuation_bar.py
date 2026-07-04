#!/usr/bin/env python3
"""
Generate per-class average fluctuation bar chart for Baseline vs Adversarial model.
Data from measure_victim_fluctuation.py (2000 members + 2000 non-members, averaged).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLASSES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
    'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural\nThickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding',
]

# ─── Real data from measure_victim_fluctuation.py ────────────────────────────
# Adversarial model: victim_adversarial_eps002_noaug.pth (avg of member + non-member)
adv_fluct = [
    (0.1992 + 0.2058) / 2,  # Atelectasis
    (0.1180 + 0.1025) / 2,  # Consolidation
    (0.2050 + 0.1970) / 2,  # Infiltration
    (0.0268 + 0.0279) / 2,  # Pneumothorax
    (0.0166 + 0.0175) / 2,  # Edema
    (0.0137 + 0.0131) / 2,  # Emphysema
    (0.0161 + 0.0202) / 2,  # Fibrosis
    (0.0868 + 0.0848) / 2,  # Effusion
    (0.0102 + 0.0120) / 2,  # Pneumonia
    (0.0203 + 0.0201) / 2,  # Pleural_Thickening
    (0.0196 + 0.0195) / 2,  # Cardiomegaly
    (0.0716 + 0.0782) / 2,  # Nodule
    (0.0396 + 0.0410) / 2,  # Mass
    (0.0008 + 0.0007) / 2,  # Hernia
    (0.3800 + 0.3814) / 2,  # No Finding
]

# Baseline model: victim_baseline.pth (avg of member + non-member)
base_fluct = [
    (0.2536 + 0.2516) / 2,  # Atelectasis
    (0.1193 + 0.1131) / 2,  # Consolidation
    (0.1311 + 0.1328) / 2,  # Infiltration
    (0.0917 + 0.0952) / 2,  # Pneumothorax
    (0.0781 + 0.0715) / 2,  # Edema
    (0.0579 + 0.0557) / 2,  # Emphysema
    (0.0675 + 0.0710) / 2,  # Fibrosis
    (0.1246 + 0.1224) / 2,  # Effusion
    (0.0644 + 0.0645) / 2,  # Pneumonia
    (0.1648 + 0.1611) / 2,  # Pleural_Thickening
    (0.0740 + 0.0747) / 2,  # Cardiomegaly
    (0.2101 + 0.2022) / 2,  # Nodule
    (0.1665 + 0.1619) / 2,  # Mass
    (0.0101 + 0.0092) / 2,  # Hernia
    (0.1501 + 0.1476) / 2,  # No Finding
]

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})

COLOR_BASELINE = '#2196F3'
COLOR_ADVERSARIAL = '#FF5722'

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(CLASSES))
width = 0.38

bars_base = ax.bar(x - width/2, base_fluct, width, label='Baseline Model',
                   color=COLOR_BASELINE, alpha=0.85, edgecolor='white', linewidth=0.8)
bars_adv  = ax.bar(x + width/2, adv_fluct, width, label='Adversarial Model (ε=0.02)',
                   color=COLOR_ADVERSARIAL, alpha=0.85, edgecolor='white', linewidth=0.8)

# Annotate averages
avg_base = np.mean(base_fluct)
avg_adv  = np.mean(adv_fluct)
ax.axhline(y=avg_base, color=COLOR_BASELINE, linestyle='--', linewidth=1.2, alpha=0.6)
ax.axhline(y=avg_adv, color=COLOR_ADVERSARIAL, linestyle='--', linewidth=1.2, alpha=0.6)
ax.text(len(CLASSES) - 0.5, avg_base + 0.005, f'Baseline avg: {avg_base:.3f}',
        fontsize=9, color=COLOR_BASELINE, ha='right', fontstyle='italic')
ax.text(len(CLASSES) - 0.5, avg_adv + 0.005, f'Adversarial avg: {avg_adv:.3f}',
        fontsize=9, color=COLOR_ADVERSARIAL, ha='right', fontstyle='italic')

ax.set_xlabel('Disease Class')
ax.set_ylabel('Average Fluctuation  |f(x) − f(x_adv)|')
ax.set_title('Per-Class Average Output Fluctuation Under FGSM Perturbation (ε = 0.01)')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha='right')
ax.set_ylim(0, 0.45)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('fluctuation_per_class.png', dpi=300, bbox_inches='tight')
print("Saved: fluctuation_per_class.png")
plt.close(fig)
