
import numpy as np
from collections import Counter

# ---------------------------------------
# PITCH / HARMONY FEATURES
# ---------------------------------------

def pitch_class_histogram(notes, chord_root=0):
    """
    notes: [(start, end, midi_note, velocity), ...]
    chord_root: int 0-11 (C=0, C#=1, ..., B=11)
    return: np.array shape (12,) - relative to chord_root
    """

    pcs = [((n - chord_root) % 12) for (_, _, n, _) in notes]
    hist = np.zeros(12)

    for pc in pcs:
        hist[pc] += 1

    if hist.sum() > 0:
        hist /= hist.sum()  # normalize

    return hist


def chord_root_feature(bar_chord):
    """
    bar_chord: 'C:maj', 'F#:min', 'E:pcset', etc.
    return: one-hot root (12,) + mode (3,)
    """

    roots = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    root_vec = np.zeros(12)
    mode_vec = np.zeros(3)  # maj / min / other

    if bar_chord:
        root, mode = bar_chord.split(":")
        if root in roots:
            root_vec[roots.index(root)] = 1

        if mode == "maj":
            mode_vec[0] = 1
        elif mode == "min":
            mode_vec[1] = 1
        else:
            mode_vec[2] = 1

    return np.concatenate([root_vec, mode_vec])


# ---------------------------------------
# RHYTHM FEATURES
# ---------------------------------------

def rhythm_density(notes, segment_start, segment_end):
    """
    notes: all notes
    segment_start / end: tick boundaries
    """

    seg_notes = [
        n for n in notes
        if segment_start <= n[0] < segment_end
    ]

    duration = max(1, segment_end - segment_start)
    return np.array([len(seg_notes) / duration])


def onset_position_histogram(notes, segment_start, segment_end, bins=8):
    """
    Histogram of note onsets inside segment
    """

    hist = np.zeros(bins)
    length = segment_end - segment_start

    if length <= 0:
        return hist

    for (st, _, _, _) in notes:
        if segment_start <= st < segment_end:
            pos = (st - segment_start) / length
            idx = min(int(pos * bins), bins - 1)
            hist[idx] += 1

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


# ---------------------------------------
# STRUCTURAL FEATURES
# ---------------------------------------

def structural_position_features(
    bar_index,
    phrase_index,
    bars_per_phrase=4
):
    """
    Encode position inside phrase
    """

    pos_in_phrase = (bar_index - 1) % bars_per_phrase
    norm_pos = pos_in_phrase / bars_per_phrase

    return np.array([
        bar_index,
        phrase_index,
        norm_pos
    ])


# ---------------------------------------
# MAIN FEATURE VECTOR
# ---------------------------------------

def build_feature_vector(
    notes,
    segment_start,
    segment_end,
    bar_chord,
    bar_index,
    phrase_index
):
    """
    Assemble all features into ONE vector
    """
    # Извлекаем chord_root из bar_chord
    chord_root = 0  # default C
    if bar_chord:
        root, _ = bar_chord.split(":")
        roots = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        if root in roots:
            chord_root = roots.index(root)

    pitch = pitch_class_histogram(notes, chord_root)  # передаем chord_root
    chord = chord_root_feature(bar_chord)
    density = rhythm_density(notes, segment_start, segment_end)
    rhythm_hist = onset_position_histogram(
        notes, segment_start, segment_end
    )
    structure = structural_position_features(
        bar_index, phrase_index
    )

    return np.concatenate([
        pitch,
        chord,
        density,
        rhythm_hist,
        structure
    ])
