import mido

mid = mido.MidiFile("data/annotated/Ludovico Einaudi — Fly.mid")

for i, track in enumerate(mid.tracks):
    print(f"\n=== TRACK {i}: {track.name if hasattr(track, 'name') else ''} ===")
    t = 0
    for msg in track:
        t += msg.time
        if msg.is_meta:
            print(f"{t:8d} | {msg}")