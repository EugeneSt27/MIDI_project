import mido
from mido import MetaMessage, MidiTrack
from collections import defaultdict


# -------------------------------------------------
# ВСПОМОГАТЕЛЬНОЕ
# -------------------------------------------------

def abs_to_delta(events):
    """[(abs_time, msg)] → msgs with delta time"""
    events.sort(key=lambda x: x[0])
    out = []
    prev = 0
    for t, msg in events:
        msg.time = t - prev
        out.append(msg)
        prev = t
    return out


# -------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# -------------------------------------------------

def annotate_midi(
    mid,
    bars,
    phrases,
    bar_chords,
    key,
    ticks_per_beat
):
    """
    bars        : {bar_idx: [beat_indices]}
    phrases     : {phrase_idx: [bar_indices]}
    bar_chords  : {bar_idx: chord_label}
    key         : 'C:maj'
    """

    annotated = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    annotated.tracks = [track.copy() for track in mid.tracks]  # копируем ВСЕ исходные треки

    # ---------- TRACK: STRUCTURE ----------
    structure_track = MidiTrack()
    structure_track.append(MetaMessage("track_name", name="STRUCTURE", time=0))

    events = []

    # такты
    for bar, beats in bars.items():
        t = int(beats[0] * ticks_per_beat)
        events.append((t, MetaMessage("marker", text=f"BAR_{bar}")))

    # фразы
    for phr, bars_ in phrases.items():
        first_bar = min(bars_)
        first_beat = bars[first_bar][0]
        t = int(first_beat * ticks_per_beat)
        events.append((t, MetaMessage("marker", text=f"PHRASE_{phr}")))

    structure_track.extend(abs_to_delta(events))
    annotated.tracks.append(structure_track)

    # ---------- TRACK: HARMONY ----------
    harmony_track = MidiTrack()
    harmony_track.append(MetaMessage("track_name", name="HARMONY", time=0))

    events = []
    for bar, chord in bar_chords.items():
        beat = bars[bar][0]
        t = int(beat * ticks_per_beat)
        events.append((t, MetaMessage("marker", text=f"CHORD:{chord}")))

    harmony_track.extend(abs_to_delta(events))
    annotated.tracks.append(harmony_track)

    # ---------- TRACK: KEY ----------
    key_track = MidiTrack()
    key_track.append(MetaMessage("track_name", name="KEY", time=0))
    tonic, mode = key.split(":")

    key_track.append(MetaMessage("key_signature", key=tonic.lower() if mode == "min" else tonic, time=0))
    annotated.tracks.append(key_track)

    return annotated
