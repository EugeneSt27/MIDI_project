"""
features.py — извлечение признаков для каждого бара.

Каждый бар описывается тремя независимыми компонентами (профиль бара):

    "harmony" — тональный вектор (14 элементов)
        Представляет аккорд в тональном пространстве.
        Не просто one-hot, а позиция на квинтовом круге + терцовые соотношения.
        Благодаря этому C:maj и G:maj будут близки (квинта), а C:maj и C:min —
        умеренно близки (параллельные), что музыкально корректно.

    "melody" — мелодический профиль (24 элемента)
        12: относительная pitch-class гистограмма (ноты относительно корня аккорда)
        12: интервальная гистограмма (распределение интервалов между соседними нотами)
        Интервальная часть улавливает контур мелодии (прыжки vs плавное движение)
        независимо от транспозиции.

    "rhythm" — ритмический профиль (10 элементов)
        8: гистограмма позиций онсетов (равномерно ли распределены ноты в баре)
        1: плотность нот (нормализованная логарифмически)
        1: синкопированность (доля нот не на сильных долях)

Итого: 3 вектора, 48 элементов суммарно.
Веса убраны — каждое измерение сравнивается отдельно.
"""

import numpy as np

PCN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Позиции нот на квинтовом круге (C=0, G=1, D=2, ... F=11)
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


# ---------------------------------------
# UTILS
# ---------------------------------------

def notes_in_segment(notes, segment_start, segment_end):
    return [
        n for n in notes
        if n[0] < segment_end and n[1] > segment_start
    ]


def _parse_chord(bar_chord):
    """Returns (root_str, mode_str) or (None, None) if invalid/empty."""
    if not bar_chord or ":" not in bar_chord:
        return None, None
    root, mode = bar_chord.split(":", 1)
    return root, mode


# =======================================
# HARMONY COMPONENT (14 dims)
# =======================================

def tonal_vector(bar_chord):
    """
    Представляет аккорд в тональном пространстве через квинтовый круг.

    Каждый аккорд — это трезвучие (набор нот). Каждая нота кодируется
    через sin/cos своей позиции на квинтовом круге. Вектор = сумма
    этих проекций + one-hot мажор/минор.

    Это даёт осмысленные расстояния: C:maj близко к G:maj (квинта),
    далеко от F#:min. Гораздо лучше чем one-hot корня.

    Размер: 12 (tonal projection) + 2 (mode: maj/min) = 14
    """
    tonal = np.zeros(12)
    mode_vec = np.zeros(2)  # [maj, min]

    root, mode = _parse_chord(bar_chord)
    if root is None or root not in PCN:
        return np.concatenate([tonal, mode_vec])

    root_idx = PCN.index(root)

    if mode == "maj":
        chord_notes = [root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]
        mode_vec[0] = 1.0
    elif mode == "min":
        chord_notes = [root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]
        mode_vec[1] = 1.0
    else:
        chord_notes = [root_idx]

    for note in chord_notes:
        angle = 2 * np.pi * CIRCLE_OF_FIFTHS[note] / 12
        tonal[note] += np.cos(angle)
        tonal[(note + 6) % 12] += np.sin(angle)

    if np.linalg.norm(tonal) > 0:
        tonal /= np.linalg.norm(tonal)

    return np.concatenate([tonal, mode_vec])  # 14


# =======================================
# MELODY COMPONENT (24 dims)
# =======================================

def relative_pitch_histogram(notes, chord_root):
    """
    Pitch-class гистограмма относительно корня аккорда.
    Показывает какие ступени лада используются (инвариантно к транспозиции).
    """
    hist = np.zeros(12)
    for (_, _, pitch, _) in notes:
        pc = (pitch - chord_root) % 12
        hist[pc] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def interval_histogram(notes):
    """
    Распределение интервалов между соседними нотами (по времени начала).
    Интервалы 0..11 полутонов (модуль октавы).

    Улавливает контур мелодии: плавное движение (много малых секунд)
    vs прыжки (большие интервалы). Полностью инвариантно к транспозиции.
    """
    hist = np.zeros(12)
    if len(notes) < 2:
        return hist

    sorted_notes = sorted(notes, key=lambda n: n[0])
    for i in range(1, len(sorted_notes)):
        interval = abs(sorted_notes[i][2] - sorted_notes[i-1][2]) % 12
        hist[interval] += 1

    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def melody_profile(notes, chord_root):
    """
    Мелодический профиль = rel_pitch (12) + interval_hist (12) = 24 dims.
    """
    rel = relative_pitch_histogram(notes, chord_root)
    ivl = interval_histogram(notes)
    return np.concatenate([rel, ivl])


# =======================================
# RHYTHM COMPONENT (10 dims)
# =======================================

def onset_histogram(notes, segment_start, segment_end, bins=8):
    """
    Гистограмма позиций онсетов внутри бара (8 бинов).
    Показывает ритмический паттерн независимо от высоты нот.
    """
    hist = np.zeros(bins)
    length = segment_end - segment_start
    if length <= 0:
        return hist
    for (st, _, _, _) in notes:
        pos = (st - segment_start) / length
        idx = min(max(int(pos * bins), 0), bins - 1)
        hist[idx] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def note_density(notes, segment_start, segment_end):
    """
    Нормализованная плотность нот (логарифмическая шкала).
    """
    duration = max(1, segment_end - segment_start)
    raw = len(notes) / duration
    return np.array([np.log1p(raw * 100) / np.log1p(100)])


def syncopation_score(notes, segment_start, segment_end, beats_per_bar=4):
    """
    Синкопированность: доля нот которые начинаются НЕ на сильных долях.
    Высокое → синкопированный ритм. Низкое → ноты строго на долях.
    """
    length = segment_end - segment_start
    if length <= 0 or not notes:
        return np.array([0.0])

    beat_duration = length / beats_per_bar
    tolerance = beat_duration * 0.1

    off_beat = 0
    for (st, _, _, _) in notes:
        pos = st - segment_start
        nearest_beat = round(pos / beat_duration) * beat_duration
        if abs(pos - nearest_beat) > tolerance:
            off_beat += 1

    return np.array([off_beat / len(notes)])


def rhythm_profile(notes, segment_start, segment_end):
    """
    Ритмический профиль = onset_hist (8) + density (1) + syncopation (1) = 10 dims.
    """
    oh = onset_histogram(notes, segment_start, segment_end)
    nd = note_density(notes, segment_start, segment_end)
    sc = syncopation_score(notes, segment_start, segment_end)
    return np.concatenate([oh, nd, sc])


# =======================================
# MAIN: BAR PROFILE
# =======================================

def build_bar_profile(notes, segment_start, segment_end, bar_chord):
    """
    Строит профиль бара — словарь из трёх независимых компонентов.

    Возвращает:
        {
            "harmony": np.array (14,),
            "melody":  np.array (24,),
            "rhythm":  np.array (10,),
        }

    Каждый компонент сравнивается отдельно — нет единого взвешенного вектора.
    Это позволяет находить паттерны типа:
        harmony=high, melody=low, rhythm=high  →  "перемелодизация"
        harmony=low,  melody=high, rhythm=high →  "перегармонизация"
    """
    seg_notes = notes_in_segment(notes, segment_start, segment_end)

    chord_root = 0
    root, _ = _parse_chord(bar_chord)
    if root is not None and root in PCN:
        chord_root = PCN.index(root)

    return {
        "harmony": tonal_vector(bar_chord),
        "melody":  melody_profile(seg_notes, chord_root),
        "rhythm":  rhythm_profile(seg_notes, segment_start, segment_end),
    }


# =======================================
# BACKWARD COMPATIBILITY
# =======================================

def build_feature_vector(notes, segment_start, segment_end, bar_chord):
    """
    Плоский вектор для совместимости с similarity_matrix.py и graph_inheritance.py.
    Порядок: harmony (14) + melody (24) + rhythm (10) = 48 dims.
    """
    profile = build_bar_profile(notes, segment_start, segment_end, bar_chord)
    return np.concatenate([
        profile["harmony"],
        profile["melody"],
        profile["rhythm"],
    ])