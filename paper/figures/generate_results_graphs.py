#!/usr/bin/env python3
"""
Generate 3 results figures for the IEEE paper using real saved attack results.

Graph 1: ε vs AUC line plot (both models)
Graph 2: Evasion Defense comparison bar chart (AUC under attack at ε=0.02)
Graph 3: MIA Vulnerability comparison bar chart (Attack accuracy)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─── Real data from fgsm_attack/logs ─────────────────────────────────────────
# From fgsm_baseline_comparison.md and fgsm_adversarial_comparison.md
epsilons            = [0.0,  0.001,  0.002,  0.005,  0.01,   0.02]
baseline_auc        = [0.8278, 0.6832, 0.5842, 0.4396, 0.3591, 0.3110]
adversarial_auc     = [0.7516, 0.5115, 0.4669, 0.7935, 0.9044, 0.8264]
baseline_flip       = [0.0,  0.0744, 0.1287, 0.2052, 0.2419, 0.2612]
adversarial_flip    = [0.0,  0.0422, 0.0504, 0.0188, 0.0540, 0.0387]

# From Fluctuation_MIA/logs/attack_results.txt and attack_results_overfit.txt
# Best classifier (Gradient Boosting) for each
mia_baseline_acc      = 0.5016
mia_baseline_var_acc  = 0.5004
mia_adv_acc           = 0.6730
mia_adv_var_acc       = 0.6744

# ─── Style constants ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})

COLOR_BASELINE = '#2196F3'      # Blue
COLOR_ADVERSARIAL = '#FF5722'   # Deep Orange
COLOR_MIA_BASE = '#42A5F5'      # Light Blue
COLOR_MIA_ADV  = '#EF5350'      # Light Red
RANDOM_LINE = '#888888'

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: ε vs Flip Rate Line Plot
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

ax1.plot(epsilons, baseline_flip, 'o-', color=COLOR_BASELINE, linewidth=2.2,
         markersize=8, label='Baseline Model', zorder=5)
ax1.plot(epsilons, adversarial_flip, 's--', color=COLOR_ADVERSARIAL, linewidth=2.2,
         markersize=8, label='Adversarial Model (ε=0.02)', zorder=5)

# Annotate key points
for i in range(1, len(epsilons)):
    ax1.annotate(f'{baseline_flip[i]:.1%}', (epsilons[i], baseline_flip[i]),
                 textcoords="offset points", xytext=(0, 10), fontsize=9,
                 color=COLOR_BASELINE, ha='center', fontweight='bold')
    ax1.annotate(f'{adversarial_flip[i]:.1%}', (epsilons[i], adversarial_flip[i]),
                 textcoords="offset points", xytext=(0, -16), fontsize=9,
                 color=COLOR_ADVERSARIAL, ha='center', fontweight='bold')

ax1.set_xlabel('FGSM Perturbation Magnitude (ε)')
ax1.set_ylabel('Flip Rate (fraction of flipped predictions)')
ax1.set_title('Evasion Attack: Prediction Flip Rate vs. ε')
ax1.set_ylim(-0.01, 0.32)
ax1.set_xticks(epsilons)
ax1.set_xticklabels(['0\n(clean)', '0.001', '0.002', '0.005', '0.01', '0.02'])
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fig1.tight_layout()
fig1.savefig('epsilon_vs_fliprate.png', dpi=300, bbox_inches='tight')
print("Saved: epsilon_vs_fliprate.png")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Evasion Defense Bar Chart (Flip Rate at different ε levels)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

# Only show the attacked epsilons (no clean, flip rate is 0)
eps_labels_bar = ['ε=0.001', 'ε=0.002', 'ε=0.005', 'ε=0.01', 'ε=0.02']
x = np.arange(len(eps_labels_bar))
width = 0.35

bars1 = ax2.bar(x - width/2, baseline_flip[1:], width, label='Baseline Model',
                color=COLOR_BASELINE, alpha=0.85, edgecolor='white', linewidth=0.8)
bars2 = ax2.bar(x + width/2, adversarial_flip[1:], width, label='Adversarial Model',
                color=COLOR_ADVERSARIAL, alpha=0.85, edgecolor='white', linewidth=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9, color=COLOR_BASELINE, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
             f'{height:.1%}', ha='center', va='bottom', fontsize=9, color=COLOR_ADVERSARIAL, fontweight='bold')

ax2.set_xlabel('FGSM Perturbation Magnitude')
ax2.set_ylabel('Flip Rate')
ax2.set_title('Evasion Attack: Prediction Flip Rate Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(eps_labels_bar)
ax2.set_ylim(0, 0.32)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig2.tight_layout()
fig2.savefig('evasion_defense_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: evasion_defense_comparison.png")
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 3: Robustness–Privacy Tradeoff (Evasion vs MIA side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(9, 4.5))

models = ['Baseline\nModel', 'Adversarial\nModel']

# Left panel: Evasion Vulnerability (Flip Rate at ε=0.02) — lower = better defense
evasion_vals = [baseline_flip[-1] * 100, adversarial_flip[-1] * 100]  # Flip rate at ε=0.02, as %
colors_evasion = [COLOR_BASELINE, COLOR_ADVERSARIAL]
bars_ev = ax3a.bar(models, evasion_vals, color=colors_evasion, alpha=0.85,
                   width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars_ev, evasion_vals):
    ax3a.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
              f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3a.set_ylabel('Flip Rate at ε=0.02 (%)')
ax3a.set_title('Evasion Vulnerability', fontweight='bold')
ax3a.set_ylim(0, 35)
ax3a.grid(True, axis='y', alpha=0.3)
ax3a.spines['top'].set_visible(False)
ax3a.spines['right'].set_visible(False)
# Add arrow annotation
ax3a.annotate('', xy=(1, adversarial_flip[-1] * 100 + 1), xytext=(0, baseline_flip[-1] * 100 - 1),
              arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax3a.text(0.5, 0.55, '↓ More\nRobust', ha='center', fontsize=9, color='green', fontstyle='italic',
          transform=ax3a.transAxes)

# Right panel: MIA Vulnerability
mia_vals = [mia_adv_var_acc * 100, mia_adv_var_acc * 100]  # Will use correct values below
mia_vals = [mia_baseline_acc * 100, mia_adv_var_acc * 100]
colors_mia = [COLOR_BASELINE, COLOR_ADVERSARIAL]
bars_mia = ax3b.bar(models, mia_vals, color=colors_mia, alpha=0.85,
                    width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars_mia, mia_vals):
    ax3b.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
              f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3b.axhline(y=50, color=RANDOM_LINE, linestyle=':', linewidth=1.5, alpha=0.9, label='Random guessing (50%)')
ax3b.set_ylabel('MIA Accuracy (%)')
ax3b.set_title('Privacy Vulnerability', fontweight='bold')
ax3b.set_ylim(0, 85)
ax3b.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax3b.grid(True, axis='y', alpha=0.3)
ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
# Add arrow annotation
ax3b.annotate('', xy=(1, mia_adv_var_acc * 100 - 1), xytext=(0, mia_baseline_acc * 100 + 1),
              arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3b.text(0.5, 0.72, '↑ More\nVulnerable', ha='center', fontsize=9, color='red', fontstyle='italic',
          transform=ax3b.transAxes)

fig3.suptitle('The Robustness–Privacy Tradeoff', fontsize=16, fontweight='bold', y=1.02)
fig3.tight_layout()
fig3.savefig('tradeoff_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: tradeoff_comparison.png")
plt.close(fig3)

print("\nAll 3 graphs generated successfully!")
