import math
from collections import defaultdict

def ticks_to_beats(notes, ticks_per_beat):
    """
    notes: [(start_tick, end_tick, pitch, velocity)]
    returns: sorted list of beat indices
    """
    beats = set()
    for st, _, _, _ in notes:
        beat = int(st // ticks_per_beat)
        beats.add(beat)
    return sorted(beats)

def beats_to_bars(beats, ts=(4,4)):
    beats_per_bar = ts[0] * (4 / ts[1])
    bars = defaultdict(list)
    for b in beats:
        bar = int(b // beats_per_bar) + 1
        bars[bar].append(b)
    return bars

def bars_to_phrases(bars, bars_per_phrase=4):
    phrases = defaultdict(list)
    for bar in bars:
        phrase = (bar - 1) // bars_per_phrase + 1
        phrases[phrase].append(bar)
    return phrases
