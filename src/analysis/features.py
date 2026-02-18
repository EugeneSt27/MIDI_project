import numpy as np


# ---------------------------------------
# UTILS
# ---------------------------------------

PCN = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def notes_in_segment(notes, segment_start, segment_end):
    """
    Filter notes that overlap with a segment.
    """
    return [
        n for n in notes
        if n[0] < segment_end and n[1] > segment_start
    ]


# ---------------------------------------
# PITCH FEATURES
# ---------------------------------------

def absolute_pitch_class_histogram(notes):
    """
    Absolute pitch-class histogram (no harmonic context).
    """
    hist = np.zeros(12)

    for (_, _, pitch, _) in notes:
        hist[pitch % 12] += 1

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


def relative_pitch_class_histogram(notes, chord_root):
    """
    Pitch-class histogram relative to chord root.
    Encodes functional role of material.
    """
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

def chord_root_feature(bar_chord):
    """
    bar_chord: 'C:maj', 'F#:min', 'E:pcset', etc.
    return: one-hot root (12,) + mode (3,)
    """

    root_vec = np.zeros(12)
    mode_vec = np.zeros(3)  # maj / min / other

    if bar_chord:
        root, mode = bar_chord.split(":")

        if root in PCN:
            root_vec[PCN.index(root)] = 1

        if mode == "maj":
            mode_vec[0] = 1
        elif mode == "min":
            mode_vec[1] = 1
        else:
            mode_vec[2] = 1

    return np.concatenate([root_vec, mode_vec])


def harmonic_complexity_feature(bar_chord):
    """
    Rough indicator of harmonic stability / ambiguity.
    """
    if bar_chord is None:
        return np.array([0.0])

    if bar_chord.endswith("pcset") or bar_chord == "N":
        return np.array([1.0])

    return np.array([0.0])


# ---------------------------------------
# RHYTHM FEATURES
# ---------------------------------------

def rhythm_density(notes, segment_start, segment_end):
    """
    Note density inside a segment.
    """
    duration = max(1, segment_end - segment_start)
    return np.array([len(notes) / duration])


def onset_position_histogram(notes, segment_start, segment_end, bins=8):
    """
    Histogram of note onset positions inside a segment.
    """
    hist = np.zeros(bins)
    length = segment_end - segment_start

    if length <= 0:
        return hist

    for (st, _, _, _) in notes:
        pos = (st - segment_start) / length
        idx = min(int(pos * bins), bins - 1)
        hist[idx] += 1

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


def note_count_feature(notes):
    """
    Normalized note count (very rough material activity measure).
    """
    return np.array([len(notes)])


# ---------------------------------------
# MAIN FEATURE VECTOR
# ---------------------------------------

def build_feature_vector(
    notes,
    segment_start,
    segment_end,
    bar_chord
):
    """
    Build feature vector for a single BAR (segment).
    """

    # restrict notes to this bar
    seg_notes = notes_in_segment(notes, segment_start, segment_end)

    # extract chord root
    chord_root = 0  # default C
    if bar_chord:
        root, _ = bar_chord.split(":")
        if root in PCN:
            chord_root = PCN.index(root)

    # pitch features
    abs_pitch = absolute_pitch_class_histogram(seg_notes)
    rel_pitch = relative_pitch_class_histogram(seg_notes, chord_root)

    # harmony features
    chord_feat = chord_root_feature(bar_chord)
    harm_complex = harmonic_complexity_feature(bar_chord)

    # rhythm features
    density = rhythm_density(seg_notes, segment_start, segment_end)
    rhythm_hist = onset_position_histogram(
        seg_notes, segment_start, segment_end
    )

    # material activity
    note_count = note_count_feature(seg_notes)

    return np.concatenate([
        rel_pitch,        # 12
        abs_pitch,        # 12
        chord_feat,       # 15
        harm_complex,     # 1
        density,          # 1
        rhythm_hist,      # 8
        note_count        # 1
    ])
