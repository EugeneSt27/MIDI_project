import mido
from mido import Message, MetaMessage, MidiTrack
from collections import defaultdict

# -------------------------------------------------

ANNOTATION_RANGES = {
    "BAR":     range(0, 32),     # 0–31
    "PHRASE":  range(32, 64),    # 32–63
    "CHORD":   range(64, 96),    # 64–95
}

CHORD_QUALITY_TO_INT = {
    "maj": 0,
    "min": 1,
    "pcset": 2,
    "N": 3,
}

ANNOTATION_CHANNEL = 15
ANNOTATION_VELOCITY = 1
ANNOTATION_DURATION = 1  # tick

def make_annotation_note(note, t):
    assert 0 <= note <= 127
    return [
        (t, Message(
            "note_on",
            note=note,
            velocity=ANNOTATION_VELOCITY,
            channel=ANNOTATION_CHANNEL
        )),
        (t + ANNOTATION_DURATION, Message(
            "note_off",
            note=note,
            velocity=0,
            channel=ANNOTATION_CHANNEL
        )),
    ]


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
    annotated = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    annotated.tracks = [track.copy() for track in mid.tracks]

    # =================================================
    # TRACK: ANNOTATION_STRUCTURE
    # =================================================
    structure_track = MidiTrack()
    structure_track.append(
        MetaMessage("track_name", name="ANNOTATION_STRUCTURE", time=0)
    )

    events = []

    # ---- BAR ----
    for bar_idx, beats in bars.items():
        t = int(beats[0] * ticks_per_beat)

        note = ANNOTATION_RANGES["BAR"].start + (bar_idx % len(ANNOTATION_RANGES["BAR"]))
        assert note in ANNOTATION_RANGES["BAR"]

        events += make_annotation_note(note, t)

    # ---- PHRASE ----
    for phrase_idx, bar_list in phrases.items():
        first_bar = min(bar_list)
        t = int(bars[first_bar][0] * ticks_per_beat)

        note = (
            ANNOTATION_RANGES["PHRASE"].start
            + (phrase_idx % len(ANNOTATION_RANGES["PHRASE"]))
        )
        assert note in ANNOTATION_RANGES["PHRASE"]

        events += make_annotation_note(note, t)

    structure_track.extend(abs_to_delta(events))
    annotated.tracks.append(structure_track)

    # =================================================
    # TRACK: ANNOTATION_HARMONY
    # =================================================
    harmony_track = MidiTrack()
    harmony_track.append(
        MetaMessage("track_name", name="ANNOTATION_HARMONY", time=0)
    )

    events = []

    for bar_idx, chord_label in bar_chords.items():
        beat = bars[bar_idx][0]
        t = int(beat * ticks_per_beat)

        # chord_label ожидается вида "X:quality" или "N"
        if chord_label == "N":
            quality = "N"
        else:
            _, quality = chord_label.split(":")

        if quality not in CHORD_QUALITY_TO_INT:
            continue  # неизвестный класс — игнорируем

        q_code = CHORD_QUALITY_TO_INT[quality]

        note = ANNOTATION_RANGES["CHORD"].start + q_code
        assert note in ANNOTATION_RANGES["CHORD"]

        events += make_annotation_note(note, t)

    harmony_track.extend(abs_to_delta(events))
    annotated.tracks.append(harmony_track)

    # =================================================
    # TRACK: ANNOTATION_KEY (без нот)
    # =================================================
    key_track = MidiTrack()
    key_track.append(
        MetaMessage("track_name", name="ANNOTATION_KEY", time=0)
    )

    tonic, mode = key.split(":")
    key_track.append(
        MetaMessage(
            "key_signature",
            key=tonic.lower() if mode == "min" else tonic,
            time=0
        )
    )

    annotated.tracks.append(key_track)

    return annotated
