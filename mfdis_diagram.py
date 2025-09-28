import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

class MFDISVisualizer:
    def __init__(self):
        self.colors = {
            'blue': '#A8D5E5',
            'green': '#B8E6B3',
            'arrow': '#2C3E50',
            'text': '#1A237E',
            'highlight': '#4CAF50'
        }
        
    def create_node(self, ax, x, y, width, height, title, subtitle=None, color='blue', alpha=0.9):
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x - width/2, y - height/2),
            width, height,
            boxstyle=f"round,pad=0.3,rounding_size=0.2",
            fc=self.colors[color],
            ec=self.colors['arrow'],
            alpha=alpha,
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add title
        ax.text(x, y + height/4, title,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=self.colors['text'])
                
        if subtitle:
            ax.text(x, y - height/4, subtitle,
                    ha='center', va='center',
                    fontsize=8, color=self.colors['text'],
                    style='italic')
                    
        return box

    def create_curved_arrow(self, ax, start, end, direction='right', color=None, style='-'):
        if color is None:
            color = self.colors['arrow']
            
        connectionstyle = f"arc3,rad={0.3 if direction=='right' else -0.3}"
        arrow = patches.FancyArrowPatch(
            start, end,
            connectionstyle=connectionstyle,
            arrowstyle='->',
            color=color,
            linewidth=2,
            linestyle=style
        )
        ax.add_patch(arrow)
        return arrow

    def add_icon_text(self, ax, x, y, text, fontsize=8):
        ax.text(x, y, text,
                ha='center', va='center',
                fontsize=fontsize,
                color=self.colors['text'])

    def create_decentralized_network(self):
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        plt.title("MFDIS - Decentralized Intelligence & Adaptive Network",
                 pad=20, fontsize=16, fontweight='bold', color=self.colors['text'])

        # Create main components
        components = {
            'data_capture': {
                'pos': (0.2, 0.7),
                'title': "1. Data Capture\n(ASHA/ANM)",
                'subtitle': "Mobile App Input\n(Images, Text, Voice)",
                'color': 'blue'
            },
            'local_ai': {
                'pos': (0.5, 0.7),
                'title': "2. Local AI Node\n(PHC/Clinic)",
                'subtitle': "Case Management,\nSpecialist Orchestration",
                'color': 'green'
            },
            'cloud': {
                'pos': (0.8, 0.7),
                'title': "Cloud & Specialists\n(Central)",
                'subtitle': "Case Management,\nReview, Orchestration",
                'color': 'blue'
            }
        }

        # Add main components
        for key, comp in components.items():
            self.create_node(ax, comp['pos'][0], comp['pos'][1], 0.25, 0.15,
                           comp['title'], comp['subtitle'], comp['color'])

        # Add Federated Learning Loop
        fl_center = (0.5, 0.4)
        circle = plt.Circle(fl_center, 0.15, fill=False,
                          linestyle='--', color=self.colors['arrow'], linewidth=2)
        ax.add_artist(circle)
        
        self.add_icon_text(ax, fl_center[0], fl_center[1] - 0.05,
                          "4. Federated Learning Loop\nContinuous AI Improvement\n(Model Weights Only)")

        # Add connections
        # Main flow
        self.create_curved_arrow(ax, (0.325, 0.7), (0.375, 0.7))
        self.add_icon_text(ax, 0.35, 0.75, "Encrypted Data")

        self.create_curved_arrow(ax, (0.625, 0.7), (0.675, 0.7))
        self.add_icon_text(ax, 0.65, 0.75, "Draft Report +")

        # Federated learning arrows
        self.create_curved_arrow(ax, (0.5, 0.55), (0.5, 0.4), 'left', style='--')
        self.create_curved_arrow(ax, (0.5, 0.4), (0.5, 0.55), 'right', style='--')

        # Add feature labels at the top
        features_top = [
            (0.2, 0.9, "Scalable & Modular"),
            (0.5, 0.9, "Data Security & Privacy"),
            (0.8, 0.9, "Final Diagnosis & Privacy")
        ]
        
        for x, y, text in features_top:
            self.add_icon_text(ax, x, y, text, 10)

        # Add feature labels at the bottom
        features_bottom = [
            (0.2, 0.2, "Offline Capable"),
            (0.5, 0.2, "Continuous AI Improvement"),
            (0.8, 0.2, "Nationwide Reach")
        ]
        
        for x, y, text in features_bottom:
            self.add_icon_text(ax, x, y, text, 10)

        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Save the figure
        plt.savefig('mfdis_decentralized_network.png', 
                    bbox_inches='tight',
                    dpi=300,
                    facecolor='white',
                    edgecolor='none')
        plt.close()

if __name__ == "__main__":
    visualizer = MFDISVisualizer()
    visualizer.create_decentralized_network()
    print("MFDIS Decentralized Network diagram has been generated as 'mfdis_decentralized_network.png'")