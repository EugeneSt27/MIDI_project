import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np
from src.analysis.development import find_parents
from src.config import WEIGHTS
import plotly.graph_objects as go


# ---------------------------------------
# BUILD GRAPH
# ---------------------------------------

def build_knowledge_graph(
    segments: Dict[int, List[int]],
    features: Dict[int, np.ndarray],
    similarity_threshold: float = 0.75,
    max_parents: int = 3,
    weights: Dict[str, float] = None
) -> nx.DiGraph:
    """
    segments: {phrase_id: [bar_ids]}
    features: {segment_id: feature_vector}  # segment_id = phrase_id
    weights: {'pitch': 1, 'chord': 1.5, 'rhythm': 2, 'struct': 1}
    
    return:
        directed graph (knowledge graph) with inheritance edges
    """
    if weights is None:
        weights = WEIGHTS
    
    # Получаем edges с многомерным сходством
    edges = find_parents(features, similarity_threshold, max_parents, weights)
    
    G = nx.DiGraph()
    
    for parent, child, weight in edges:
        G.add_node(parent, label=f"Phrase {parent}")
        G.add_node(child, label=f"Phrase {child}")
        G.add_edge(parent, child, weight=round(weight, 3))
    
    return G


# ---------------------------------------
# VISUALIZE GRAPH
# ---------------------------------------

def draw_graph(G: nx.DiGraph, title="Knowledge Graph", save_path=None):
    """
    Visualize the graph
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Node colors based on in-degree
    node_colors = [G.in_degree(n) for n in G.nodes]
    
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        node_size=500,
        font_size=10,
        edge_color='gray',
        arrows=True
    )
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()


# ---------------------------------------
# BASIC ANALYSIS
# ---------------------------------------

def analyze_graph(G: nx.DiGraph) -> dict:
    """
    Returns simple interpretable statistics
    """
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "root_nodes": [n for n in G.nodes if G.in_degree(n) == 0],
        "leaf_nodes": [n for n in G.nodes if G.out_degree(n) == 0],
        "most_influential": sorted(
            G.nodes,
            key=lambda n: G.out_degree(n),
            reverse=True
        )[:5]
    }


def draw_beautiful_graph(
    G: nx.DiGraph,
    features: Dict[int, np.ndarray],
    title: str = "Beautiful Musical Knowledge Graph",
    save_path: str | None = None
):
    """
    Draws graph using Matplotlib for static, beautiful visualization.
    Nodes colored by phrase ID, sized by rhythm density, edges filtered >0.95 with categorical labels.
    """
    
    plt.figure(figsize=(14, 10))
    
    if nx.is_directed_acyclic_graph(G):  # Проверяем, DAG ли
        pos = nx.kamada_kawai_layout(G)  # Hierarchical для дерева inheritance (лучше для DAG)
    else:
        pos = nx.circular_layout(G)  # Circular для общего случая
    
    # ----- Nodes -----
    node_colors = list(G.nodes)
    node_sizes = []

    for n in G.nodes:
        density = features.get(n, np.array([0]))[27]  # rhythm density
        node_sizes.append(max(300, density * 1200))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.tab10,
        alpha=0.85
    )

    # ----- Edges -----
    edge_colors = []
    edge_labels = {}

    for u, v, d in G.edges(data=True):
        w = d['weight']
        if w >= 0.97:
            edge_colors.append('red')
            edge_labels[(u, v)] = 'strong'
        elif w >= 0.85:
            edge_colors.append('orange')
            edge_labels[(u, v)] = 'variation'
        else:
            edge_colors.append('gray')
            edge_labels[(u, v)] = 'weak'

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=2,
        arrows=True,
        arrowsize=20
    )

    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)

    plt.title(title, fontsize=16)
    plt.axis('off')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Strong similarity'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Variation'),
        plt.Line2D([0], [0], color='gray', lw=2, label='Weak inheritance'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()