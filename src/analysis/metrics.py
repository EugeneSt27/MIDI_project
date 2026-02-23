"""
metrics.py — метрики для анализа структуры музыкального произведения.

Все метрики принимают либо SSM (np.ndarray shape [N, N]),
либо список рёбер графа наследования [(parent, child, weight), ...].

Группы метрик:
    A. SSM — блочность, периодичность, контраст, связность
    B. Graph — покрытие, дальние связи, концентрация наследования
    C. Combined — итоговая оценка структурной организованности
"""

import numpy as np
from typing import Dict, List, Tuple


# ================================================================
# A. SSM METRICS
# ================================================================

def block_score(sim_matrix: np.ndarray, window: int = 8) -> float:
    """
    Блочная структура: насколько матрица организована в секции вдоль диагонали.

    Считает разницу между средним сходством ВНУТРИ диагонального окна
    и средним сходством ВНЕ окна.

    Высокий результат (~0.2 и выше) → чёткие секции (как у "5 fly").
    Низкий (~0.0) → нет секций, однородная или хаотичная структура.
    """
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
    """
    Периодичность: есть ли регулярно повторяющийся паттерн (остинато, рефрен).

    Для каждого лага k считает среднее сходство по диагонали матрицы,
    смещённой на k. Пик автокорреляции → доминирующий период произведения.

    Возвращает (periodicity_strength, dominant_period).

    Высокий strength (~0.7+) → сильное остинато (Einaudi Dietro Casa).
    Умеренный (~0.5-0.7) → есть повторяющиеся темы (нормальная структура).
    Низкий (<0.4) → нет регулярности (генерация или через-чур свободная форма).
    """
    n = sim_matrix.shape[0]
    max_period = min(max_period, n // 2)

    lag_means = []
    for k in range(1, max_period + 1):
        diag = [sim_matrix[i, i + k] for i in range(n - k)]
        lag_means.append((k, float(np.mean(diag))))

    if not lag_means:
        return 0.0, 0

    dominant_period = max(lag_means, key=lambda x: x[1])
    strength = dominant_period[1]

    return strength, dominant_period[0]


def contrast_score(sim_matrix: np.ndarray) -> float:
    """
    Контраст: насколько широк разброс значений в матрице (без диагонали).

    Это НЕ то же самое что variance — мы смотрим на межквартильный размах
    (IQR = Q75 - Q25), который устойчив к выбросам.

    Высокий IQR → есть и очень похожие, и очень непохожие пары баров.
    Это признак развитой структуры: есть разные темы И их повторения.

    Низкий IQR → либо всё одинаково, либо всё одинаково непохоже.
    """
    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = sim_matrix[mask]

    if len(values) == 0:
        return 0.0

    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def long_range_similarity(sim_matrix: np.ndarray, min_distance: int = 8) -> float:
    """
    Дальние связи: среднее сходство между барами которые далеко друг от друга.

    Показывает насколько темы возвращаются через большие промежутки.
    В классической форме (A-B-A, рондо, фуга) этот показатель должен быть
    выше чем у генерации, где дальние бары случайно не похожи.

    Интерпретация:
        > 0.65 → сильное возвращение тем (сонатная форма, рондо)
        0.55–0.65 → умеренное (типично для хорошей музыки)
        < 0.55 → дальние бары не связаны (генерация)
    """
    n = sim_matrix.shape[0]
    values = []

    for i in range(n):
        for j in range(i + min_distance, n):
            values.append(sim_matrix[i, j])

    if not values:
        return 0.0

    return float(np.mean(values))


def diagonal_dominance(sim_matrix: np.ndarray) -> float:
    """
    Диагональное доминирование: насколько локальные связи сильнее глобальных.

    Отношение среднего по ближней диагонали (лаг 1-4)
    к среднему по всей матрице (без диагонали).

    > 1.0 → локальные бары похожи больше чем случайные (хорошо, есть плавность)
    ≈ 1.0 → нет локальной когерентности
    """
    n = sim_matrix.shape[0]
    local_vals = []
    for k in range(1, min(5, n)):
        for i in range(n - k):
            local_vals.append(sim_matrix[i, i + k])

    mask = ~np.eye(n, dtype=bool)
    global_mean = float(np.mean(sim_matrix[mask]))

    if global_mean == 0 or not local_vals:
        return 1.0

    return float(np.mean(local_vals) / global_mean)


# ================================================================
# B. GRAPH METRICS
# ================================================================

def graph_coverage(
    edges: List[Tuple[int, int, float]],
    num_bars: int
) -> float:
    """
    Покрытие графа: доля баров у которых есть хотя бы один родитель.

    Изолированные бары (без родителя) — признак того что этот бар
    не похож ни на один предыдущий. В хорошей музыке таких должно быть мало
    (обычно только вступительные бары новых тем).
    В генерации — много изолированных баров.

    1.0 → все бары связаны с предыдущими
    0.0 → ни один бар не похож на предыдущие (хаос)
    """
    if num_bars <= 1:
        return 1.0

    children = set(child for _, child, _ in edges)
    # первый бар не может иметь родителя — исключаем
    eligible = num_bars - 1
    if eligible == 0:
        return 1.0

    return len(children) / eligible


def long_range_edge_ratio(
    edges: List[Tuple[int, int, float]],
    threshold: int = 8
) -> float:
    """
    Доля рёбер которые соединяют бары далеко друг от друга (> threshold).

    В форме с возвращением тем (A-B-A) должна быть высокой —
    бар 30 наследует от бара 1, потому что тема вернулась.
    В генерации — либо очень низкая (только локальные связи),
    либо случайная.
    """
    if not edges:
        return 0.0

    long = sum(1 for p, c, _ in edges if abs(c - p) > threshold)
    return long / len(edges)


def inheritance_concentration(
    edges: List[Tuple[int, int, float]],
    num_bars: int
) -> float:
    """
    Концентрация наследования: насколько неравномерно распределены родители.

    Считает нормализованный коэффициент Джини по числу детей у каждого бара.

    Высокий Джини (~0.6+) → есть несколько "архетипных" баров от которых
    наследует большинство. Это признак тематического единства.

    Низкий Джини (~0.0-0.3) → наследование размазано равномерно или отсутствует.
    Характерно для генерации: нет доминирующей темы.
    """
    if not edges or num_bars == 0:
        return 0.0

    parent_counts = np.zeros(num_bars)
    for parent, _, _ in edges:
        parent_counts[parent] += 1

    # коэффициент Джини
    counts = np.sort(parent_counts)
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * counts) / (n * counts.sum())) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def mean_edge_weight(edges: List[Tuple[int, int, float]]) -> float:
    """
    Среднее сходство по всем рёбрам графа.

    Показывает насколько "уверенны" связи наследования.
    Высокое значение → бары которые наследуют друг от друга действительно похожи.
    """
    if not edges:
        return 0.0
    return float(np.mean([w for _, _, w in edges]))


# ================================================================
# C. COMBINED
# ================================================================

def structure_score(
    sim_matrix: np.ndarray,
    edges: List[Tuple[int, int, float]]
) -> float:
    """
    Итоговая оценка структурной организованности произведения [0..1].

    Взвешенная комбинация ключевых метрик:
        - block_score (блочность секций)
        - contrast_score (разнообразие)
        - graph_coverage (связность)
        - inheritance_concentration (тематическое единство)

    Это экспериментальная метрика — веса можно подбирать.
    Интерпретация: чем выше, тем более организована структура.
    """
    n = sim_matrix.shape[0]

    bs = block_score(sim_matrix)
    cs = contrast_score(sim_matrix)
    gc = graph_coverage(edges, n)
    ic = inheritance_concentration(edges, n)

    # нормализуем block_score (типичный диапазон 0..0.3) → 0..1
    bs_norm = np.clip(bs / 0.3, 0.0, 1.0)
    # contrast_score (типичный диапазон 0..0.4) → 0..1
    cs_norm = np.clip(cs / 0.4, 0.0, 1.0)

    score = 0.35 * bs_norm + 0.25 * cs_norm + 0.25 * gc + 0.15 * ic
    return float(score)


# ================================================================
# MAIN ENTRY POINT
# ================================================================

def evaluate_track(
    similarity_matrix: np.ndarray,
    graph_edges: List[Tuple[int, int, float]],
    window: int = 8,
    long_range_min_distance: int = 8,
    long_range_edge_threshold: int = 8,
) -> Dict[str, float]:
    """
    Полная оценка трека. Возвращает словарь всех метрик.

    SSM метрики:
        block_score              — блочность секций (выше = чище секции)
        periodicity_strength     — сила периодического паттерна
        dominant_period          — период в барах (2=остинато, 8=фраза и т.д.)
        contrast_score           — IQR сходств (разнообразие материала)
        long_range_similarity    — среднее сходство далёких баров
        diagonal_dominance       — локальные связи vs глобальные

    Graph метрики:
        graph_coverage           — доля баров с родителем
        long_range_edge_ratio    — доля дальних рёбер
        inheritance_concentration — Джини по числу детей (тематическое единство)
        mean_edge_weight         — среднее сходство по рёбрам

    Combined:
        structure_score          — итоговая оценка организованности [0..1]
    """
    n = similarity_matrix.shape[0]

    ps, dp = periodicity_score(similarity_matrix)

    return {
        # --- SSM ---
        "block_score":           block_score(similarity_matrix, window),
        "periodicity_strength":  ps,
        "dominant_period":       float(dp),
        "contrast_score":        contrast_score(similarity_matrix),
        "long_range_similarity": long_range_similarity(similarity_matrix, long_range_min_distance),
        "diagonal_dominance":    diagonal_dominance(similarity_matrix),
        # --- Graph ---
        "graph_coverage":              graph_coverage(graph_edges, n),
        "long_range_edge_ratio":       long_range_edge_ratio(graph_edges, long_range_edge_threshold),
        "inheritance_concentration":   inheritance_concentration(graph_edges, n),
        "mean_edge_weight":            mean_edge_weight(graph_edges),
        # --- Combined ---
        "structure_score":       structure_score(similarity_matrix, graph_edges),
    }