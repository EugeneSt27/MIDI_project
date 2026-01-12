from collections import defaultdict
import numpy as np

PCN = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ---------- META EVENTS ----------

def merge_tempo_changes(tracks):
    events = []
    for tr in tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == 'set_tempo':
                events.append((t, msg.tempo))
    events.sort()
    return events or [(0, 500000)]

def time_signatures(tracks):
    events = []
    for tr in tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == 'time_signature':
                events.append((t, (msg.numerator, msg.denominator)))
    events.sort()
    return events or [(0, (4, 4))]

def key_signatures(tracks):
    events = []
    for tr in tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == 'key_signature':
                events.append((t, msg.key))
    return sorted(events)

# ---------- NOTES ----------

def extract_notes(mid):
    notes = []
    for tr in mid.tracks:
        t = 0
        active = defaultdict(list)
        for msg in tr:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note].append((t, msg.velocity))
            elif msg.type in ('note_off', 'note_on'):
                if active[msg.note]:
                    st, vel = active[msg.note].pop()
                    notes.append((st, t, msg.note, vel))
    return notes
