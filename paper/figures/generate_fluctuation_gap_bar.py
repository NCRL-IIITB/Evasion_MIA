#!/usr/bin/env python3
"""
Generate per-class fluctuation GAP (Members - Non-Members) bar chart.
Shows the membership signal: how differently the model behaves on training data vs unseen data.
Data from measure_victim_fluctuation.py (2000 members + 2000 non-members).
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
# Adversarial model (victim_adversarial_eps002_noaug.pth)
#                     Members    Non-Members
adv_members =       [0.1992, 0.1180, 0.2050, 0.0268, 0.0166, 0.0137, 0.0161, 0.0868, 0.0102, 0.0203, 0.0196, 0.0716, 0.0396, 0.0008, 0.3800]
adv_nonmembers =    [0.2058, 0.1025, 0.1970, 0.0279, 0.0175, 0.0131, 0.0202, 0.0848, 0.0120, 0.0201, 0.0195, 0.0782, 0.0410, 0.0007, 0.3814]

# Baseline model (victim_baseline.pth)
base_members =      [0.2536, 0.1193, 0.1311, 0.0917, 0.0781, 0.0579, 0.0675, 0.1246, 0.0644, 0.1648, 0.0740, 0.2101, 0.1665, 0.0101, 0.1501]
base_nonmembers =   [0.2516, 0.1131, 0.1328, 0.0952, 0.0715, 0.0557, 0.0710, 0.1224, 0.0645, 0.1611, 0.0747, 0.2022, 0.1619, 0.0092, 0.1476]

# Compute gaps (Member - Non-Member)
adv_gap  = [m - nm for m, nm in zip(adv_members, adv_nonmembers)]
base_gap = [m - nm for m, nm in zip(base_members, base_nonmembers)]

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

bars_base = ax.bar(x - width/2, base_gap, width, label='Baseline Model',
                   color=COLOR_BASELINE, alpha=0.85, edgecolor='white', linewidth=0.8)
bars_adv  = ax.bar(x + width/2, adv_gap, width, label='Adversarial Model (ε=0.02)',
                   color=COLOR_ADVERSARIAL, alpha=0.85, edgecolor='white', linewidth=0.8)

# Zero line
ax.axhline(y=0, color='black', linewidth=0.8)

# Average lines
avg_base = np.mean(base_gap)
avg_adv  = np.mean(adv_gap)
ax.axhline(y=avg_base, color=COLOR_BASELINE, linestyle='--', linewidth=1.2, alpha=0.6)
ax.axhline(y=avg_adv, color=COLOR_ADVERSARIAL, linestyle='--', linewidth=1.2, alpha=0.6)
ax.text(len(CLASSES) - 0.3, avg_base + 0.0008, f'Baseline avg: {avg_base:+.4f}',
        fontsize=9, color=COLOR_BASELINE, ha='right', fontstyle='italic')
ax.text(len(CLASSES) - 0.3, avg_adv - 0.0018, f'Adversarial avg: {avg_adv:+.4f}',
        fontsize=9, color=COLOR_ADVERSARIAL, ha='right', fontstyle='italic')

ax.set_xlabel('Disease Class')
ax.set_ylabel('Fluctuation Gap  (Members − Non-Members)')
ax.set_title('Per-Class Fluctuation Gap: Members vs. Non-Members (ε = 0.01)')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha='right')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('fluctuation_gap_members_vs_nonmembers.png', dpi=300, bbox_inches='tight')
print("Saved: fluctuation_gap_members_vs_nonmembers.png")
plt.close(fig)
