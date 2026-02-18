import numpy as np
from typing import Dict
from sklearn.cluster import DBSCAN

from .base import semantic_similarity


def cluster_bars_dbscan(
    feature_vectors: Dict[int, np.ndarray],
    weights: Dict[str, float],
    eps: float = 0.3,
    min_samples: int = 2
) -> Dict[int, int]:
    """
    DBSCAN clustering based on semantic distance:
    distance = 1 - similarity
    """

    ids = sorted(feature_vectors.keys())
    n = len(ids)

    dist_matrix = np.zeros((n, n))

    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):
            sims = semantic_similarity(
                feature_vectors[id_i],
                feature_vectors[id_j],
                weights
            )
            dist_matrix[i, j] = 1.0 - sims["total"]

    model = DBSCAN(
        metric="precomputed",
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(dist_matrix)

    return {
        bar_id: int(label)
        for bar_id, label in zip(ids, labels)
    }
