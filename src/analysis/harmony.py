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


def harmony_by_bar(notes, bars, ticks_per_beat, ts=(4, 4)):
    """
    notes: [(start_tick, end_tick, pitch, velocity)]
    bars: {bar_idx: [beat_indices]}
    returns: {bar_idx: chord_label}
    bar_chords = {}
    beats_per_bar = ts[0] * (4 / ts[1])
    last_chord = "N"

    for bar, _ in bars.items():
        bar_start_tick = (bar - 1) * beats_per_bar * ticks_per_beat
        bar_end_tick = bar * beats_per_bar * ticks_per_beat

        pitches = [
            pitch
            for st, en, pitch, _ in notes
            if st < bar_end_tick and en > bar_start_tick
        ]

        chord = chord_from_pitches(pitches)
        
        # Fallback to the previous chord if the current bar has no recognized pitches ("N")
        # to avoid empty bar artifacts in SSM matrices.
        if chord == "N" and last_chord != "N":
            bar_chords[bar] = last_chord
        else:
            bar_chords[bar] = chord
            last_chord = chord

    return bar_chords
