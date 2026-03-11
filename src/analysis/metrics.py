"""
metrics.py — метрики для анализа структуры музыкального произведения.
"""

import numpy as np
from typing import Dict, List, Tuple


# ================================================================
# A. SSM METRICS
# ================================================================

def block_score(sim_matrix: np.ndarray, window: int = 8) -> float:
    n = sim_matrix.shape[0]
    if n < window * 2:
        window = max(2, n // 4)
    inside, outside = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(i - j) <= window:
                inside.append(sim_matrix[i, j])
            else:
                outside.append(sim_matrix[i, j])
    if not inside or not outside:
        return 0.0
    return float(np.mean(inside) - np.mean(outside))


def periodicity_score(sim_matrix: np.ndarray, max_period: int = 16) -> Tuple[float, int]:
    n = sim_matrix.shape[0]
    max_period = min(max_period, n // 2)
    lag_means = []
    for k in range(1, max_period + 1):
        diag = [sim_matrix[i, i + k] for i in range(n - k)]
        lag_means.append((k, float(np.mean(diag))))
    if not lag_means:
        return 0.0, 0
    dominant_period = max(lag_means, key=lambda x: x[1])
    return dominant_period[1], dominant_period[0]


def contrast_score(sim_matrix: np.ndarray) -> float:
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = sim_matrix[mask]
    if len(values) == 0:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def long_range_similarity(sim_matrix: np.ndarray, min_distance: int = 8) -> float:
    n = sim_matrix.shape[0]
    values = [sim_matrix[i, j] for i in range(n) for j in range(i + min_distance, n)]
    return float(np.mean(values)) if values else 0.0


def diagonal_dominance(sim_matrix: np.ndarray) -> float:
    n = sim_matrix.shape[0]
    local_vals = [sim_matrix[i, i + k] for k in range(1, min(5, n)) for i in range(n - k)]
    mask = ~np.eye(n, dtype=bool)
    global_mean = float(np.mean(sim_matrix[mask]))
    if global_mean == 0 or not local_vals:
        return 1.0
    return float(np.mean(local_vals) / global_mean)


# ================================================================
# B. GRAPH METRICS
# ================================================================

def graph_coverage(edges: List[Tuple[int, int, float]], num_bars: int) -> float:
    if num_bars <= 1:
        return 1.0
    children = set(child for _, child, _ in edges)
    eligible = num_bars - 1
    return len(children) / eligible if eligible > 0 else 1.0


def long_range_edge_ratio(edges: List[Tuple[int, int, float]], threshold: int = 8) -> float:
    if not edges:
        return 0.0
    return sum(1 for p, c, _ in edges if abs(c - p) > threshold) / len(edges)


def inheritance_concentration(edges: List[Tuple[int, int, float]], num_bars: int) -> float:
    if not edges or num_bars == 0:
        return 0.0
    parent_counts = np.zeros(num_bars)
    for parent, _, _ in edges:
        parent_counts[parent] += 1
    counts = np.sort(parent_counts)
    n = len(counts)
    if counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * counts) / (n * counts.sum())) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def mean_edge_weight(edges: List[Tuple[int, int, float]]) -> float:
    if not edges:
        return 0.0
    return float(np.mean([w for _, _, w in edges]))


# ================================================================
# C. CHANGE TYPE METRICS (из phrase_analysis)
# ================================================================

def change_type_stats(phrase_analysis: Dict) -> Dict[str, float]:
    """
    Считает долю каждого типа изменений между фразами по всему треку.

    Возвращает:
        pct_repetition      — доля повторений (H H H везде)
        pct_remelodization  — гармония стабильна, мелодия меняется
        pct_reharmonization — мелодия стабильна, гармония меняется
        pct_full_change     — всё меняется
        pct_mixed           — нет доминирующего паттерна
        change_diversity    — энтропия распределения типов [0..1]
    """
    from collections import Counter

    counts = Counter()
    for sec_patterns in phrase_analysis.get("level1", {}).values():
        for p in sec_patterns:
            counts[p["change_type"]] += 1

    total = sum(counts.values())
    types = ["repetition", "remelodization", "reharmonization", "full_change", "mixed"]

    if total == 0:
        result = {f"pct_{t}": 0.0 for t in types}
        result["change_diversity"] = 0.0
        return result

    result = {f"pct_{t}": counts[t] / total for t in types}

    # энтропия, нормализованная на log2(5)
    probs = np.array([counts[t] / total for t in types])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    result["change_diversity"] = entropy / np.log2(5)

    return result


# ================================================================
# D. COMBINED
# ================================================================

def structure_score(sim_matrix: np.ndarray, edges: List[Tuple[int, int, float]]) -> float:
    n = sim_matrix.shape[0]
    bs = block_score(sim_matrix)
    cs = contrast_score(sim_matrix)
    gc = graph_coverage(edges, n)
    ic = inheritance_concentration(edges, n)
    bs_norm = np.clip(bs / 0.3, 0.0, 1.0)
    cs_norm = np.clip(cs / 0.4, 0.0, 1.0)
    return float(0.35 * bs_norm + 0.25 * cs_norm + 0.25 * gc + 0.15 * ic)


# ================================================================
# MAIN ENTRY POINT
# ================================================================

def evaluate_track(
    similarity_matrix: np.ndarray,
    graph_edges: List[Tuple[int, int, float]],
    phrase_analysis: Dict = None,
    window: int = 8,
    long_range_min_distance: int = 8,
    long_range_edge_threshold: int = 8,
) -> Dict[str, float]:

    n = similarity_matrix.shape[0]
    ps, dp = periodicity_score(similarity_matrix)

    results = {
        # --- SSM ---
        "block_score":           block_score(similarity_matrix, window),
        "periodicity_strength":  ps,
        "dominant_period":       float(dp),
        "contrast_score":        contrast_score(similarity_matrix),
        "long_range_similarity": long_range_similarity(similarity_matrix, long_range_min_distance),
        "diagonal_dominance":    diagonal_dominance(similarity_matrix),
        # --- Graph ---
        "graph_coverage":             graph_coverage(graph_edges, n),
        "long_range_edge_ratio":      long_range_edge_ratio(graph_edges, long_range_edge_threshold),
        "inheritance_concentration":  inheritance_concentration(graph_edges, n),
        "mean_edge_weight":           mean_edge_weight(graph_edges),
        # --- Combined ---
        "structure_score": structure_score(similarity_matrix, graph_edges),
    }

    # --- Change type stats (если передан phrase_analysis) ---
    if phrase_analysis is not None:
        results.update(change_type_stats(phrase_analysis))

    return results