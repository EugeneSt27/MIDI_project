import numpy as np


PCN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ---------------------------------------
# UTILS
# ---------------------------------------

def notes_in_segment(notes, segment_start, segment_end):
    return [
        n for n in notes
        if n[0] < segment_end and n[1] > segment_start
    ]


# ---------------------------------------
# PITCH FEATURES
# ---------------------------------------

def absolute_pitch_class_histogram(notes):
    hist = np.zeros(12)
    for (_, _, pitch, _) in notes:
        hist[pitch % 12] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def relative_pitch_class_histogram(notes, chord_root):
    hist = np.zeros(12)
    for (_, _, pitch, _) in notes:
        pc = (pitch - chord_root) % 12
        hist[pc] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


# ---------------------------------------
# HARMONY FEATURES
# ---------------------------------------

def _parse_chord(bar_chord):
    """
    Returns (root_str, mode_str) or (None, None) if invalid / empty.
    """
    if not bar_chord or ":" not in bar_chord:
        return None, None
    root, mode = bar_chord.split(":", 1)
    return root, mode


def chord_root_feature(bar_chord):
    root_vec = np.zeros(12)
    mode_vec = np.zeros(3)   # maj / min / other

    root, mode = _parse_chord(bar_chord)
    if root is not None and root in PCN:
        root_vec[PCN.index(root)] = 1
        if mode == "maj":
            mode_vec[0] = 1
        elif mode == "min":
            mode_vec[1] = 1
        else:
            mode_vec[2] = 1

    return np.concatenate([root_vec, mode_vec])


def harmonic_complexity_feature(bar_chord):
    root, mode = _parse_chord(bar_chord)
    if root is None:
        return np.array([0.0])
    if mode in ("pcset", "N"):
        return np.array([1.0])
    return np.array([0.0])


# ---------------------------------------
# RHYTHM FEATURES
# ---------------------------------------

def rhythm_density(notes, segment_start, segment_end):
    duration = max(1, segment_end - segment_start)
    return np.array([len(notes) / duration])


def onset_position_histogram(notes, segment_start, segment_end, bins=8):
    hist = np.zeros(bins)
    length = segment_end - segment_start
    if length <= 0:
        return hist
    for (st, _, _, _) in notes:
        pos = (st - segment_start) / length
        idx = min(max(int(pos * bins), 0), bins - 1)  # было: min(int(pos * bins), bins - 1)
        hist[idx] += 1
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def note_count_feature(notes):
    return np.array([len(notes)])


# ---------------------------------------
# MAIN FEATURE VECTOR
# ---------------------------------------

def build_feature_vector(notes, segment_start, segment_end, bar_chord):
    """
    Build feature vector for a single bar.

    Layout (50 elements total):
        [0:12]   rel_pitch
        [12:24]  abs_pitch
        [24:39]  chord_feat (root 12 + mode 3)
        [39:40]  harm_complex
        [40:41]  density
        [41:49]  rhythm_hist (8 bins)
        [49:50]  note_count
    """
    seg_notes = notes_in_segment(notes, segment_start, segment_end)

    # chord root for relative pitch
    chord_root = 0
    root, _ = _parse_chord(bar_chord)
    if root is not None and root in PCN:
        chord_root = PCN.index(root)

    rel_pitch   = relative_pitch_class_histogram(seg_notes, chord_root)  # 12
    abs_pitch   = absolute_pitch_class_histogram(seg_notes)               # 12
    chord_feat  = chord_root_feature(bar_chord)                           # 15
    harm_complex = harmonic_complexity_feature(bar_chord)                 # 1
    density     = rhythm_density(seg_notes, segment_start, segment_end)   # 1
    rhythm_hist = onset_position_histogram(seg_notes, segment_start, segment_end)  # 8
    note_count  = note_count_feature(seg_notes)                           # 1

    return np.concatenate([
        rel_pitch,
        abs_pitch,
        chord_feat,
        harm_complex,
        density,
        rhythm_hist,
        note_count,
    ])  # total: 50