import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as patches
from matplotlib.path import Path

def create_mfdis_architecture():
    # Create figure for the first architecture diagram
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title("MFDIS AI DATA FLOW ARCHITECTURE", pad=20, fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Define the main components/nodes
    components = [
        ("Data\nIngestion", 0.1, 0.5, '#E3F2FD'),
        ("On-Device\nProcessing", 0.3, 0.5, '#E8F5E9'),
        ("Node-Level\nProcessing", 0.5, 0.5, '#E8F5E9'),
        ("Cloud-Level\nRouting", 0.7, 0.5, '#E3F2FD'),
        ("Specialist\nReview", 0.9, 0.5, '#E8F5E9')
    ]
    
    # Add component boxes
    for i, (label, x, y, color) in enumerate(components, 1):
        box = FancyBboxPatch((x-0.08, y-0.15), 0.16, 0.3,
                            boxstyle="round,pad=0.02",
                            fc=color, ec='#2C3E50', alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, f"{i}\n{label}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='#2C3E50')
    
    # Add arrows connecting components
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                      color='#2C3E50', lw=2)
    
    for i in range(len(components)-1):
        x1, y1 = components[i][1]+0.08, components[i][2]
        x2, y2 = components[i+1][1]-0.08, components[i+1][2]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)
    
    # Add federated learning loop
    fl_circle = plt.Circle((0.5, 0.2), 0.1, fill=False, 
                          linestyle='--', color='#2C3E50')
    ax.add_artist(fl_circle)
    ax.text(0.5, 0.1, 'Federated Learning\n& Model Improvement',
            ha='center', va='center', fontsize=8, color='#2C3E50')
    
    # Set axis properties
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('mfdis_architecture_1.png', bbox_inches='tight', dpi=300, 
                facecolor='white', edgecolor='none')
    plt.close()

def create_decentralized_network():
    # Create figure for the second architecture diagram
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title("MFDIS - Decentralized Intelligence & Adaptive Network", 
              pad=20, fontsize=14, fontweight='bold', color='#2C3E50')
    
    # Define the main components
    components = {
        'data_capture': (0.2, 0.7, "1. Data Capture\n(ASHA/ANM)", '#E3F2FD'),
        'local_ai': (0.5, 0.7, "2. Local AI Node\n(PHC/Clinic)", '#E8F5E9'),
        'cloud': (0.8, 0.7, "Cloud & Specialists\n(Central)", '#E3F2FD'),
        'federated': (0.5, 0.3, "4. Federated Learning Loop", '#E3F2FD')
    }
    
    # Add component boxes
    for pos, (x, y, label, color) in components.items():
        box = FancyBboxPatch((x-0.15, y-0.1), 0.3, 0.2,
                            boxstyle="round,pad=0.02",
                            fc=color, ec='#2C3E50', alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#2C3E50')
    
    # Add arrows and flow indicators
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                      color='#2C3E50', lw=2)
    
    # Main flow arrows
    ax.annotate('', xy=(0.35, 0.7), xytext=(0.2, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.7), xytext=(0.5, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.7), xytext=(0.65, 0.7), arrowprops=arrow_props)
    
    # Federated learning arrows
    fl_arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                         color='#2C3E50', lw=2, linestyle='--')
    ax.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.6), arrowprops=fl_arrow_props)
    ax.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.45), arrowprops=fl_arrow_props)
    
    # Add feature labels
    features = [
        (0.1, 0.9, "Scaable & Modular"),
        (0.4, 0.9, "Data Security & Privacy"),
        (0.7, 0.9, "Final Diagnosis & Privacy"),
        (0.2, 0.15, "Offline Capable"),
        (0.5, 0.15, "Continuous AI Improvement\n(Model Weights Only)"),
        (0.8, 0.15, "Nationwide Reach")
    ]
    
    for x, y, label in features:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8, color='#2C3E50', style='italic')
    
    # Set axis properties
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('mfdis_architecture_2.png', bbox_inches='tight', dpi=300,
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    # Create both architecture diagrams
    create_mfdis_architecture()
    create_decentralized_network()
    print("Architecture diagrams have been generated as 'mfdis_architecture_1.png' and 'mfdis_architecture_2.png'")