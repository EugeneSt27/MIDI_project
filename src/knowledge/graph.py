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
    title: str = "Beautiful Musical Knowledge Graph",
    save_path: str | None = None
):
    """
    Draws graph using Plotly for interactive, beautiful visualization.
    Nodes colored by in-degree, sized by out-degree, edges by weight.
    """
    # Positions using spring layout
    pos = nx.spring_layout(G, seed=42, k=1.0, iterations=100)
    
    # Node data
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    node_text = [f"Phrase {n}<br>In-degree: {G.in_degree(n)}<br>Out-degree: {G.out_degree(n)}" for n in G.nodes]
    node_color = [G.in_degree(n) for n in G.nodes]  # Color by in-degree
    node_size = [G.out_degree(n) * 20 + 20 for n in G.nodes]  # Size by out-degree
    
    # Edge data
    edge_x = []
    edge_y = []
    edge_text = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"Weight: {d['weight']:.2f}")
    
    # Create figure
    fig = go.Figure()
    
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Edges'
    ))
    
    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"S{n}" for n in G.nodes],
        textposition="top center",
        hoverinfo='text',
        textfont=dict(size=12),
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="In-degree"),
            line=dict(width=2, color='black')
        ),
        name='Nodes'
    ))
    
    # Layout
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    if save_path:
        fig.write_html(save_path)  # Save as interactive HTML
    else:
        fig.show()  # Show in browser
