"""
base.py — функции сравнения баров.

Два режима:

1. profile_similarity(profile_a, profile_b)
   Принимает словари {"harmony": ..., "melody": ..., "rhythm": ...}
   Возвращает словарь сходств по каждому компоненту.
   Это основной режим — без весов, каждое измерение независимо.

2. semantic_similarity(vec_a, vec_b, weights)
   Принимает плоские векторы (48 dims) и словарь весов.
   Оставлен для обратной совместимости с graph_inheritance.py и similarity_matrix.py.
   Layout плоского вектора: harmony(14) + melody(24) + rhythm(10).
"""

import numpy as np
from typing import Dict


# ---------------------------------------
# COSINE
# ---------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------
# PROFILE-BASED SIMILARITY (основной)
# ---------------------------------------

def profile_similarity(
    profile_a: Dict[str, np.ndarray],
    profile_b: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Сравнивает два профиля бара по каждому компоненту отдельно.

    Возвращает:
        {
            "harmony": float,   # 0..1
            "melody":  float,   # 0..1
            "rhythm":  float,   # 0..1
        }

    Никаких весов — каждое измерение независимо.
    Это позволяет находить паттерны типа:
        harmony=0.9, melody=0.3, rhythm=0.9  →  "перемелодизация"
        harmony=0.3, melody=0.9, rhythm=0.9  →  "перегармонизация"
    """
    return {
        key: cosine_similarity(profile_a[key], profile_b[key])
        for key in profile_a
        if key in profile_b
    }


# ---------------------------------------
# FLAT VECTOR SPLITTING (для совместимости)
# ---------------------------------------

def split_feature_vector(vec: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Разбивает плоский вектор (48 dims) на компоненты.
    Layout: harmony(14) + melody(24) + rhythm(10)
    """
    return {
        "harmony": vec[0:14],
        "melody":  vec[14:38],
        "rhythm":  vec[38:48],
    }


# ---------------------------------------
# WEIGHTED SIMILARITY (для совместимости)
# ---------------------------------------

def semantic_similarity(
    a: np.ndarray,
    b: np.ndarray,
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Взвешенное сходство по плоским векторам.
    Оставлен для совместимости с graph_inheritance.py и similarity_matrix.py.

    Если weights содержит "harmony"/"melody"/"rhythm" — используются они.
    Ключ "total" добавляется автоматически.
    """
    a_parts = split_feature_vector(a)
    b_parts = split_feature_vector(b)

    sims = {}
    total = 0.0
    weight_sum = 0.0

    for key, w in weights.items():
        if key in a_parts:
            sim = cosine_similarity(a_parts[key], b_parts[key])
            sims[key] = sim
            total += sim * w
            weight_sum += w

    sims["total"] = total / weight_sum if weight_sum > 0 else 0.0
    return sims