from typing import Dict, List, Tuple
import numpy as np

from .base import semantic_similarity


def find_inheritance_edges(
    feature_vectors: Dict[int, np.ndarray],
    weights: Dict[str, float],
    similarity_threshold: float = 0.75,
    max_parents: int = 3
) -> List[Tuple[int, int, float]]:
    """
    Directed inheritance graph:
    parent -> child
    """

    edges = []
    segment_ids = sorted(feature_vectors.keys())

    for i, child_id in enumerate(segment_ids):
        child_vec = feature_vectors[child_id]
        candidates = []

        for parent_id in segment_ids[:i]:
            parent_vec = feature_vectors[parent_id]

            sims = semantic_similarity(
                child_vec, parent_vec, weights
            )

            if sims["total"] >= similarity_threshold:
                candidates.append(
                    (parent_id, child_id, sims["total"])
                )

        candidates.sort(key=lambda x: x[2], reverse=True)
        edges.extend(candidates[:max_parents])

    return edges
