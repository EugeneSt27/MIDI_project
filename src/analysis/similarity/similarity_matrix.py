import numpy as np
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(
    feature_vectors: Dict[int, np.ndarray],
    weights: Dict[str, float]
) -> Dict[str, np.ndarray]:

    ids = sorted(feature_vectors.keys())
    if not ids:
        return {}
        
    matrix = np.array([feature_vectors[i] for i in ids])

    parts = {
        "harmony": matrix[:, 0:14],
        "melody":  matrix[:, 14:38],
        "rhythm":  matrix[:, 38:48]
    }

    n = len(ids)
    matrices = {}
    total = np.zeros((n, n))
    weight_sum = 0.0

    for key, w in weights.items():
        if key in parts:
            sim = cosine_similarity(parts[key])
            # Округляем артефакты плавающей точки до [0, 1]
            sim = np.clip(sim, 0.0, 1.0)
            matrices[key] = sim
            total += sim * w
            weight_sum += w

    matrices["total"] = total / weight_sum if weight_sum > 0 else total
    return matrices
