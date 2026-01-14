import numpy as np
from typing import List, Dict, Tuple
from src.config import WEIGHTS


# ---------------------------------------
# SIMILARITY
# ---------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Standard cosine similarity between two vectors
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------
# FIND PARENTS (INHERITANCE)
# ---------------------------------------

def find_parents(
    feature_vectors: Dict[int, np.ndarray],
    similarity_threshold: float = 0.75,
    max_parents: int = 3,
    weights: Dict[str, float] = None
) -> List[Tuple[int, int, float]]:
    """
    feature_vectors:
        {segment_id: feature_vector}

    weights: {'pitch': 1, 'chord': 1.5, 'rhythm': 2, 'struct': 1}

    return:
        list of (parent_id, child_id, similarity)
    """
    if weights is None:
        weights = WEIGHTS

    edges = []

    segment_ids = sorted(feature_vectors.keys())

    for i, child_id in enumerate(segment_ids):
        child_vec = feature_vectors[child_id]

        # Разделим вектор на компоненты
        child_pitch = child_vec[:12]
        child_chord = child_vec[12:27]
        child_rhythm = child_vec[27:36]  # density + onset_hist
        child_struct = child_vec[36:]

        similarities = []

        # compare ONLY with previous segments
        for parent_id in segment_ids[:i]:
            parent_vec = feature_vectors[parent_id]
            parent_pitch = parent_vec[:12]
            parent_chord = parent_vec[12:27]
            parent_rhythm = parent_vec[27:36]
            parent_struct = parent_vec[36:]

            # Вычислим отдельные схожести
            sim_pitch = cosine_similarity(child_pitch, parent_pitch)
            sim_chord = cosine_similarity(child_chord, parent_chord)
            sim_rhythm = cosine_similarity(child_rhythm, parent_rhythm)
            sim_struct = cosine_similarity(child_struct, parent_struct)

            # Комбинированная схожесть с весами
            combined_sim = (
                sim_pitch * weights['pitch'] +
                sim_chord * weights['chord'] +
                sim_rhythm * weights['rhythm'] +
                sim_struct * weights['struct']
            ) / sum(weights.values())

            if combined_sim >= similarity_threshold:
                similarities.append((parent_id, combined_sim))

        # keep top-N parents
        similarities.sort(key=lambda x: x[1], reverse=True)
        for parent_id, sim in similarities[:max_parents]:
            edges.append((parent_id, child_id, sim))

    return edges
