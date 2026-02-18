import numpy as np
from typing import Dict, List, Tuple


# ============================================================
# SSM METRICS
# ============================================================

def ssm_metrics(sim_matrix: np.ndarray) -> Dict[str, float]:
    """
    Computes:
        - mean_off_diagonal
        - variance_off_diagonal
    """

    n = sim_matrix.shape[0]

    if n <= 1:
        return {
            "mean_similarity": 0.0,
            "variance_similarity": 0.0
        }

    mask = ~np.eye(n, dtype=bool)
    values = sim_matrix[mask]

    return {
        "mean_similarity": float(np.mean(values)),
        "variance_similarity": float(np.var(values))
    }


# ============================================================
# CLUSTERING METRICS
# ============================================================

def clustering_metrics(labels: Dict[int, int]) -> Dict[str, float]:
    """
    labels: {bar_id: cluster_label}
    """

    if len(labels) == 0:
        return {
            "repeat_ratio": 0.0,
            "largest_cluster_ratio": 0.0,
            "noise_ratio": 0.0
        }

    cluster_counts = {}
    noise_count = 0

    for label in labels.values():
        if label == -1:
            noise_count += 1
        else:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

    n = len(labels)

    # repeat ratio: bars in clusters of size > 1
    repeat_bars = sum(
        size for size in cluster_counts.values()
        if size > 1
    )

    largest_cluster = max(cluster_counts.values()) if cluster_counts else 0

    return {
        "repeat_ratio": repeat_bars / n,
        "largest_cluster_ratio": largest_cluster / n,
        "noise_ratio": noise_count / n
    }


# ============================================================
# GRAPH METRICS
# ============================================================

def graph_metrics(
    edges: List[Tuple[int, int, float]],
    num_nodes: int,
    long_range_threshold: int = 4
) -> Dict[str, float]:
    """
    edges: list of (parent, child, similarity)
    num_nodes: total number of bars
    """

    if num_nodes <= 1:
        return {
            "edge_density": 0.0,
            "long_range_ratio": 0.0
        }

    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    edge_count = len(edges)

    edge_density = edge_count / max_possible_edges

    if edge_count == 0:
        return {
            "edge_density": 0.0,
            "long_range_ratio": 0.0
        }

    long_range_edges = 0

    for parent, child, _ in edges:
        if abs(child - parent) > long_range_threshold:
            long_range_edges += 1

    long_range_ratio = long_range_edges / edge_count

    return {
        "edge_density": edge_density,
        "long_range_ratio": long_range_ratio
    }


# ============================================================
# COMBINED EVALUATION
# ============================================================

def evaluate_track(
    similarity_matrix: np.ndarray,
    #clustering_labels: Dict[int, int],
    graph_edges: List[Tuple[int, int, float]]
) -> Dict[str, float]:
    """
    Returns all 7 agreed metrics.
    """

    results = {}

    # --- SSM ---
    ssm_res = ssm_metrics(similarity_matrix)
    results.update(ssm_res)

    # --- Clustering ---
   # cluster_res = clustering_metrics(clustering_labels)
   # results.update(cluster_res)

    # --- Graph ---
    num_nodes = similarity_matrix.shape[0]
    graph_res = graph_metrics(graph_edges, num_nodes)
    results.update(graph_res)

    return results
