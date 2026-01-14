import mido

mid = mido.MidiFile("data/annotated/Ludovico Einaudi — Fly.mid")

ANNOTATION_CHANNEL = 15

for i, track in enumerate(mid.tracks):
    print(f"\n=== TRACK {i}: {track.name if hasattr(track, 'name') else ''} ===")
    t = 0
    for msg in track:
        t += msg.time

        if msg.type in ("note_on", "note_off") and msg.channel == ANNOTATION_CHANNEL:
            print(f"{t:8d} | {msg}")

