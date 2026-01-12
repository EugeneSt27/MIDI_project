# src/analysis/harmony.py

from collections import Counter

PCN = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def chord_from_pitches(pitches):
    """
    pitches: list of MIDI pitches in a bar
    returns: string chord label
    """
    pcs = set(p % 12 for p in pitches)
    if not pcs:
        return "N"

    for root in range(12):
        if {root, (root+4)%12, (root+7)%12}.issubset(pcs):
            return f"{PCN[root]}:maj"
        if {root, (root+3)%12, (root+7)%12}.issubset(pcs):
            return f"{PCN[root]}:min"

    return f"{PCN[min(pcs)]}:pcset"


def harmony_by_bar(notes, bars, ticks_per_beat):
    """
    notes: [(start_tick, end_tick, pitch, velocity)]
    bars: {bar_idx: [beat_indices]}
    returns: {bar_idx: chord_label}
    """
    bar_chords = {}

    for bar, beats in bars.items():
        bar_start_tick = beats[0] * ticks_per_beat
        bar_end_tick = (beats[-1] + 1) * ticks_per_beat

        pitches = [
            pitch
            for st, en, pitch, _ in notes
            if st < bar_end_tick and en > bar_start_tick
        ]

        bar_chords[bar] = chord_from_pitches(pitches)

    return bar_chords
