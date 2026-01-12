# src/analysis/development.py

import numpy as np
from math import sqrt


# -----------------------------
# ВСПОМОГАТЕЛЬНОЕ
# -----------------------------

def numeric_vector(feat):
    """
    Превращает признаки фразы в числовой вектор
    """
    return np.array([
        feat["mean_pitch"],
        feat["pitch_std"],
        feat["pitch_class_entropy"],
        feat["mean_onset_density"],
    ], dtype=float)


def euclidean(a, b):
    return sqrt(((a - b) ** 2).sum())


def chord_similarity(h1, h2):
    """
    Jaccard similarity по множеству аккордов
    """
    s1 = set(h1.keys())
    s2 = set(h2.keys())

    if not s1 and not s2:
        return 1.0

    return len(s1 & s2) / len(s1 | s2)


# -----------------------------
# ОСНОВНАЯ ЛОГИКА
# -----------------------------

def compare_phrases(
    features,
    dist_repeat=5.0,
    dist_variation=12.0,
    chord_sim_threshold=0.6
):
    """
    Возвращает список отношений между фразами
    """

    relations = []
    phrase_ids = sorted(features.keys())

    vectors = {
        pid: numeric_vector(features[pid])
        for pid in phrase_ids
    }

    for i, a in enumerate(phrase_ids):
        for b in phrase_ids[i + 1:]:
            fa = features[a]
            fb = features[b]

            # расстояние по числам
            d = euclidean(vectors[a], vectors[b])

            # сходство гармонии
            ch_sim = chord_similarity(
                fa["chord_histogram"],
                fb["chord_histogram"]
            )

            # -----------------------------
            # ПРАВИЛА
            # -----------------------------

            if d < dist_repeat and ch_sim > 0.8:
                rel = "repeat"

            elif d < dist_variation and ch_sim > chord_sim_threshold:
                rel = "variation"

            else:
                rel = "contrast"

            relations.append({
                "from": a,
                "to": b,
                "distance": round(d, 3),
                "chord_similarity": round(ch_sim, 3),
                "relation": rel
            })

    return relations

