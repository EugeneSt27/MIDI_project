
import numpy as np
from collections import Counter, defaultdict
import math

def extract_phrase_features(
    phrase_id,
    phrase_bars,
    bars,
    notes,
    bar_chords,
    ticks_per_beat
):
    """
    Возвращает dict с числовыми признаками фразы
    """

    # -----------------------------
    # 1. Границы фразы в тиках
    # -----------------------------
    first_bar = min(phrase_bars)
    last_bar = max(phrase_bars)

    first_beat = bars[first_bar][0]
    last_beat = bars[last_bar][-1] + 1

    start_tick = first_beat * ticks_per_beat
    end_tick = last_beat * ticks_per_beat

    # -----------------------------
    # 2. Ноты внутри фразы
    # -----------------------------
    phrase_notes = [
        (st, en, p)
        for (st, en, p, _, _) in notes
        if st < end_tick and en > start_tick
    ]

    pitches = [p for (_, _, p) in phrase_notes]

    # -----------------------------
    # 3. Pitch-based features
    # -----------------------------
    if pitches:
        mean_pitch = float(np.mean(pitches))
        pitch_std = float(np.std(pitches))

        pcs = [p % 12 for p in pitches]
        pc_counts = Counter(pcs)
        total = sum(pc_counts.values())

        entropy = 0.0
        for c in pc_counts.values():
            p = c / total
            entropy -= p * math.log2(p)

    else:
        mean_pitch = 0.0
        pitch_std = 0.0
        entropy = 0.0

    # -----------------------------
    # 4. Rhythm features
    # -----------------------------
    onset_count = len(phrase_notes)
    duration_beats = (end_tick - start_tick) / ticks_per_beat

    mean_onset_density = (
        onset_count / duration_beats if duration_beats > 0 else 0.0
    )

    # -----------------------------
    # 5. Harmony features
    # -----------------------------
    chords = [
        bar_chords[b]
        for b in phrase_bars
        if b in bar_chords
    ]

    chord_hist = dict(Counter(chords))

    # -----------------------------
    # 6. Итог
    # -----------------------------
    return {
        "phrase_id": phrase_id,
        "num_bars": len(phrase_bars),
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "pitch_class_entropy": entropy,
        "mean_onset_density": mean_onset_density,
        "chord_histogram": chord_hist,
    }


def extract_all_phrase_features(
    phrases,
    bars,
    notes,
    bar_chords,
    ticks_per_beat
):
    """
    Обрабатывает все фразы
    """
    features = {}

    for pid, phrase_bars in phrases.items():
        features[pid] = extract_phrase_features(
            pid,
            phrase_bars,
            bars,
            notes,
            bar_chords,
            ticks_per_beat
        )

    return features
