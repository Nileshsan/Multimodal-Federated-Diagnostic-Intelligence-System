import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
import matplotlib.patheffects as pe

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# --- Central Orchestrator (Cloud) ---
cloud = FancyBboxPatch((4.1, 8.5), 1.8, 1,
                       boxstyle="round,pad=0.4",
                       linewidth=2,
                       facecolor='#6DA9E4',
                       edgecolor='black')
ax.add_patch(cloud)
ax.text(5, 9, "Central\nOrchestrator", ha='center', va='center', fontsize=10, fontweight='bold')

# --- Local Clinics and PHC Node Boxes ---
node_style = dict(boxstyle="round,pad=0.3", fc="#A8D5BA", ec="black", lw=2)

# Bottom Node - PHC
ax.text(5, 1, "Local PHC/c\nNode", ha='center', va='center', fontsize=10, bbox=node_style)

# Left Node - Clinic
ax.text(1.5, 5, "Local\nClinic", ha='center', va='center', fontsize=10, bbox=node_style)

# Right Node - Clinic
ax.text(8.5, 5, "Local\nClinic", ha='center', va='center', fontsize=10, bbox=node_style)

# --- Arrows between nodes (cycle) ---
arrow_style = dict(arrowstyle="-|>", color='skyblue', lw=3)

# Clinic (left) to PHC
arrow1 = FancyArrowPatch((2.8, 5), (4.3, 2), **arrow_style)
ax.add_patch(arrow1)

# PHC to Clinic (right)
arrow2 = FancyArrowPatch((5.7, 2), (7.2, 5), **arrow_style)
ax.add_patch(arrow2)

# Clinic (right) to Orchestrator
arrow3 = FancyArrowPatch((8.5, 5.8), (6, 8.3), **arrow_style)
ax.add_patch(arrow3)

# Orchestrator to Clinic (left)
arrow4 = FancyArrowPatch((4, 8.3), (2, 5.8), **arrow_style)
ax.add_patch(arrow4)

# --- Text Annotations ---
ax.text(2, 8.3, "Send Model Weights\n(Updates ONLY)", fontsize=9, ha='left')
ax.text(7.2, 8.3, "Receive Improved\nGlobal Model", fontsize=9, ha='left')

# --- Accuracy label on side ---
ax.text(10, 5, "Accuracy (%)", fontsize=10, rotation=90, va='center')

plt.tight_layout()
plt.show()