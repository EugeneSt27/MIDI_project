from typing import Dict, List, Tuple
import numpy as np

# NOTE: import from the analysis.similarity package
# If running from project root with src/ in sys.path:
from analysis.similarity.base import semantic_similarity


def find_inheritance_edges(
    feature_vectors: Dict[int, np.ndarray],
    weights: Dict[str, float],
    similarity_threshold: float = 0.75,
    max_parents: int = 3,
) -> List[Tuple[int, int, float]]:
    """
    Build a directed inheritance graph over bars.

    For each bar (child), look at all earlier bars (candidates for parent).
    An edge parent -> child is added if similarity >= threshold.
    At most max_parents edges per child (highest similarity wins).

    Returns list of (parent_id, child_id, similarity_score).
    """

    edges = []
    segment_ids = sorted(feature_vectors.keys())

    for i, child_id in enumerate(segment_ids):
        child_vec = feature_vectors[child_id]
        candidates = []

        for parent_id in segment_ids[:i]:
            parent_vec = feature_vectors[parent_id]
            sims = semantic_similarity(child_vec, parent_vec, weights)

            if sims["total"] >= similarity_threshold:
                candidates.append((parent_id, child_id, sims["total"]))

        candidates.sort(key=lambda x: x[2], reverse=True)
        edges.extend(candidates[:max_parents])

    return edges