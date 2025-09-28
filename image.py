import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define nodes
nodes = [
    "Healthcare Access\nApp",
    "Login",
    "Index/Main/Home",
    "Input: Text + Image",
    "XAI: Reasoning\nClassification",
    "Local Model:\nMobileLM",
    "Image Model",
    "Transfer Protocol",
    "Context Understanding\n(LLM)",
    "Multimodal State",
    "Output:\nDisease Report",
    "Connect & Transfer\nto Specialist"
]

# Add nodes
for node in nodes:
    G.add_node(node)

# Define edges
edges = [
    ("Healthcare Access\nApp", "Login"),
    ("Login", "Index/Main/Home"),
    ("Index/Main/Home", "Input: Text + Image"),
    ("Input: Text + Image", "XAI: Reasoning\nClassification"),
    ("XAI: Reasoning\nClassification", "Local Model:\nMobileLM"),
    ("XAI: Reasoning\nClassification", "Image Model"),
    ("Local Model:\nMobileLM", "Transfer Protocol"),
    ("Image Model", "Transfer Protocol"),
    ("Transfer Protocol", "Context Understanding\n(LLM)"),
    ("Context Understanding\n(LLM)", "Multimodal State"),
    ("Multimodal State", "Output:\nDisease Report"),
    ("Output:\nDisease Report", "Connect & Transfer\nto Specialist"),
]

# Add edges
for edge in edges:
    G.add_edge(*edge)

# Layout for visualization
pos = nx.spring_layout(G, k=1.5, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=2800, node_color="#89CFF0", alpha=0.9)

# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", width=2)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="black")

# Title
plt.title("Multimodal Federated Diagnostic Intelligence System (MFDIS)", fontsize=12, weight="bold")

# Save to file
plt.tight_layout()
plt.savefig("MFDIS_Flowchart.png", dpi=300)
plt.show()
