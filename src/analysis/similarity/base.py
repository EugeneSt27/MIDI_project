import numpy as np
from typing import Dict, Tuple


# -----------------------------
# COSINE
# -----------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------
# FEATURE SPLITTING
# -----------------------------

def split_feature_vector(vec: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Feature vector layout (must match build_feature_vector in features.py):

    [0:12]   rel_pitch   — pitch class histogram relative to chord root
    [12:24]  abs_pitch   — absolute pitch class histogram
    [24:39]  chord_feat  — chord root one-hot (12) + mode (3)
    [39:40]  harm_complex
    [40:41]  density
    [41:49]  rhythm_hist — onset position histogram (8 bins)
    [49:50]  note_count
    """

    return {
        "rel_pitch":    vec[0:12],
        "abs_pitch":    vec[12:24],
        "chord_feat":   vec[24:39],
        "harm_complex": vec[39:40],
        "density":      vec[40:41],
        "rhythm_hist":  vec[41:49],
        "note_count":   vec[49:50],
    }


# -----------------------------
# SEMANTIC SIMILARITY
# -----------------------------

def semantic_similarity(
    a: np.ndarray,
    b: np.ndarray,
    weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Returns per-component cosine similarity + weighted total.
    Only keys present in `weights` are computed.
    """

    a_parts = split_feature_vector(a)
    b_parts = split_feature_vector(b)

    sims = {}
    total = 0.0
    weight_sum = 0.0

    for key, w in weights.items():
        sim = cosine_similarity(a_parts[key], b_parts[key])
        sims[key] = sim
        total += sim * w
        weight_sum += w

    sims["total"] = total / weight_sum if weight_sum > 0 else 0.0
    return sims