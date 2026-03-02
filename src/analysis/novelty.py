"""
novelty.py — автоматическое нахождение структуры произведения из SSM.

Идея: в SSM граница между секциями выглядит как резкое падение сходства
вдоль диагонали. Мы это улавливаем через "novelty function" —
свёртку SSM с kernel Фосслера (шахматный паттерн).

Pipeline:
    SSM → novelty curve → peaks → boundaries → sections → phrases
"""

import numpy as np
from typing import List, Tuple, Dict


# -------------------------------------------------------
# STEP 1: NOVELTY FUNCTION
# -------------------------------------------------------

def checkerboard_kernel(size: int) -> np.ndarray:
    """
    Kernel Фосслера — шахматный паттерн размера (2*size x 2*size).
    При свёртке с SSM даёт пик там где есть граница секции.

    +1 +1 | -1 -1
    +1 +1 | -1 -1
    ------+------
    -1 -1 | +1 +1
    -1 -1 | +1 +1
    """
    kernel = np.ones((2 * size, 2 * size))
    kernel[size:, :size] = -1
    kernel[:size, size:] = -1
    return kernel


def compute_novelty_curve(
    sim_matrix: np.ndarray,
    kernel_size: int = 4,
    smooth_window: int = 3,
) -> np.ndarray:
    """
    Вычисляет novelty curve — функцию "новизны" для каждого бара.

    Высокий пик в позиции i означает что здесь начинается новая секция.

    kernel_size: размер ядра в барах (4 = смотрим на окно 8x8 вокруг диагонали)
    smooth_window: сглаживание гауссом чтобы убрать шум
    """
    n = sim_matrix.shape[0]
    k = kernel = checkerboard_kernel(kernel_size)
    ksize = 2 * kernel_size
    novelty = np.zeros(n)

    for i in range(kernel_size, n - kernel_size):
        # вырезаем окно вокруг диагональной точки (i, i)
        row_start = max(0, i - kernel_size)
        row_end   = min(n, i + kernel_size)
        col_start = max(0, i - kernel_size)
        col_end   = min(n, i + kernel_size)

        window = sim_matrix[row_start:row_end, col_start:col_end]

        # подгоняем kernel под реальный размер окна
        kr = k[:window.shape[0], :window.shape[1]]
        novelty[i] = np.sum(window * kr)

    # нормализуем в [0, 1]
    if novelty.max() > novelty.min():
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min())

    # сглаживаем
    if smooth_window > 1:
        novelty = _smooth(novelty, smooth_window)

    return novelty


def _smooth(curve: np.ndarray, window: int) -> np.ndarray:
    """Простое скользящее среднее."""
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode='same')


# -------------------------------------------------------
# STEP 2: FIND BOUNDARIES
# -------------------------------------------------------

def find_boundaries(
    novelty_curve: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 4,
) -> List[int]:
    """
    Находит позиции баров где начинаются новые секции.

    threshold: минимальная высота пика (0..1)
    min_distance: минимальное расстояние между границами в барах
                  (не бывает секций короче min_distance баров)

    Возвращает список индексов баров-границ (включая 0 и n).
    """
    n = len(novelty_curve)
    peaks = _find_peaks(novelty_curve, threshold, min_distance)

    # всегда добавляем начало и конец
    boundaries = sorted(set([0] + list(peaks) + [n]))
    return boundaries


def _find_peaks(
    curve: np.ndarray,
    threshold: float,
    min_distance: int,
) -> List[int]:
    """Находит локальные максимумы выше порога с минимальным расстоянием."""
    peaks = []
    n = len(curve)

    for i in range(1, n - 1):
        if curve[i] > threshold and curve[i] >= curve[i-1] and curve[i] >= curve[i+1]:
            # проверяем что нет более близкого пика
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
            elif curve[i] > curve[peaks[-1]]:
                # заменяем предыдущий пик если этот выше
                peaks[-1] = i

    return peaks


# -------------------------------------------------------
# STEP 3: BUILD SECTIONS AND PHRASES
# -------------------------------------------------------

def boundaries_to_sections(
    boundaries: List[int],
    bar_ids: List[int],
) -> List[List[int]]:
    """
    Преобразует список границ в список секций.
    Каждая секция — список bar_id.

    Пример:
        boundaries = [0, 8, 16, 24]
        → sections = [[bar0..bar7], [bar8..bar15], [bar16..bar23]]
    """
    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        section_bars = bar_ids[start:end]
        if section_bars:
            sections.append(section_bars)
    return sections


def split_into_phrases(
    section_bars: List[int],
    bars_per_phrase: int = 4,
) -> List[List[int]]:
    """
    Делит секцию на фразы по bars_per_phrase тактов.
    Последняя фраза может быть короче если секция не делится ровно.
    """
    phrases = []
    for i in range(0, len(section_bars), bars_per_phrase):
        phrase = section_bars[i:i + bars_per_phrase]
        if phrase:
            phrases.append(phrase)
    return phrases


# -------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------

def analyze_structure(
    sim_matrix: np.ndarray,
    bar_ids: List[int],
    kernel_size: int = 4,
    threshold: float = 0.3,
    min_section_bars: int = 4,
    bars_per_phrase: int = 4,
) -> Dict:
    """
    Полный анализ структуры из SSM.

    Возвращает:
    {
        "novelty_curve": np.ndarray,        # novelty function по барам
        "boundaries": [0, 8, 16, 32, ...],  # индексы границ секций
        "sections": [[bar_ids...], ...],    # бары по секциям
        "phrases":  [[bar_ids...], ...],    # бары по фразам (все секции)
        "num_sections": int,
        "num_phrases":  int,
    }
    """
    novelty = compute_novelty_curve(sim_matrix, kernel_size)
    boundaries = find_boundaries(novelty, threshold, min_section_bars)
    sections = boundaries_to_sections(boundaries, bar_ids)

    # разбиваем каждую секцию на фразы
    all_phrases = []
    for section in sections:
        phrases = split_into_phrases(section, bars_per_phrase)
        all_phrases.extend(phrases)

    return {
        "novelty_curve": novelty,
        "boundaries":    boundaries,
        "sections":      sections,
        "phrases":       all_phrases,
        "num_sections":  len(sections),
        "num_phrases":   len(all_phrases),
    }