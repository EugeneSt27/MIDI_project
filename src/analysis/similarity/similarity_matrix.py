import numpy as np
from typing import Dict

from .base import semantic_similarity


def compute_similarity_matrix(
    feature_vectors: Dict[int, list],
    weights: Dict[str, float]
) -> Dict[str, np.ndarray]:

    ids = sorted(feature_vectors.keys())
    n = len(ids)

    # создаём матрицы динамически по ключам весов
    matrices = {
        key: np.zeros((n, n))
        for key in weights.keys()
    }

    matrices["total"] = np.zeros((n, n))

    for i, id_i in enumerate(ids):
        for j, id_j in enumerate(ids):

            sims = semantic_similarity(
                feature_vectors[id_i],
                feature_vectors[id_j],
                weights
            )

            for key in sims:
                matrices[key][i, j] = sims[key]

    return matrices
