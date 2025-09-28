import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for compatibility
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.patches import Rectangle
import numpy as np

# Set the style for a professional medical/diagnostic look
plt.style.use('seaborn-v0_8-whitegrid')

# Data from Table (2) - Personalized / client-local evaluation for Medical Diagnostic System
epochs = [1, 10, 30, 50, 75, 100]
centralized = [52.0, 78.0, 86.5, 90.8, 91.8, 92.4]
federated_personalized = [50.0, 76.5, 85.8, 89.9, 92.3, 93.6]

# Create figure with medical theme colors
fig, ax = plt.subplots(figsize=(15, 10), dpi=200)  # Increased figure size and DPI
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

# Add padding to the plot
plt.rcParams['axes.labelpad'] = 15
plt.rcParams['xtick.major.pad'] = 10
plt.rcParams['ytick.major.pad'] = 10

# Plot with medical color scheme
ax.plot(epochs, centralized, marker='o', linestyle='-', 
        color='#2C3E50', label="Centralized Model (Global Accuracy)", 
        linewidth=3.5, markersize=12, markerfacecolor='#E74C3C',
        markeredgewidth=2, markeredgecolor='#2C3E50')
ax.plot(epochs, federated_personalized, marker='s', linestyle='--', 
        color='#27AE60', label="Federated Model (Personalized Accuracy)", 
        linewidth=3.5, markersize=12, markerfacecolor='#2ECC71',
        markeredgewidth=2, markeredgecolor='#27AE60')

# Customize axes and labels with medical theme
ax.set_xlabel("Training Epochs", fontsize=18, fontweight='bold', color='#2C3E50')
ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=18, fontweight='bold', color='#2C3E50')
ax.set_title("Medical Vision Model: Centralized vs Federated Learning Performance", 
             fontsize=22, fontweight='bold', color='#2C3E50', pad=20)

# Add subtitle
ax.text(0.5, -0.15, 'Comparison of Training Approaches in Medical Diagnostic Intelligence System',
        horizontalalignment='center', transform=ax.transAxes, 
        fontsize=16, color='#7F8C8D', style='italic')

# Increase tick label sizes and customize ticks
ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=10)
ax.tick_params(axis='both', which='minor', labelsize=16, width=1, length=5)

# Format y-axis ticks with larger numbers and percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

# Make grid more visible
ax.grid(True, linestyle='--', alpha=0.4, color='#95A5A6', linewidth=1.5)

# Enhance tick labels with bold font
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(18)

# Enhance grid and spines
ax.grid(True, linestyle='--', alpha=0.3, color='#95A5A6')
for spine in ax.spines.values():
    spine.set_color('#BDC3C7')

# Customize legend
legend = ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True,
                  loc='lower right', title='Training Methods',
                  borderpad=1.5, labelspacing=1.2)
legend.get_title().set_fontsize(16)
legend.get_title().set_fontweight('bold')

# Set y-axis limits with some padding
ax.set_ylim(35, 105)

# Add performance threshold indicator
ax.axhline(y=90, color='#E74C3C', linestyle=':', alpha=0.6, linewidth=2.5)
ax.text(5, 91.5, 'Clinical Accuracy Threshold (90%)', 
        fontsize=14, color='#E74C3C', style='italic', fontweight='bold')

# Add annotations for final accuracies
ax.annotate(f'Final: {centralized[-1]}%', 
            xy=(epochs[-1], centralized[-1]), 
            xytext=(5, 7), textcoords='offset points', 
            fontsize=14, color='#2C3E50', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='#2C3E50', alpha=0.8, pad=3))
ax.annotate(f'Final: {federated_personalized[-1]}%', 
            xy=(epochs[-1], federated_personalized[-1]), 
            xytext=(5, -20), textcoords='offset points', 
            fontsize=14, color='#27AE60', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='#27AE60', alpha=0.8, pad=3))

# Adjust layout and save
fig.tight_layout()
plt.savefig("medical_vision_model_accuracy.png", 
            bbox_inches='tight', dpi=300, 
            facecolor=fig.get_facecolor(), edgecolor='none')
print("Enhanced medical diagnostic chart saved as medical_vision_model_accuracy.png")
