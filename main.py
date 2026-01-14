import mido
#import pretty_midi
#import numpy as np

from pathlib import Path
import mido

from src.midi.extract import extract_notes, time_signatures
from src.analysis.structure import (
    ticks_to_beats,
    beats_to_bars,
    bars_to_phrases
)
from src.analysis.harmony import harmony_by_bar
from src.midi.annotate import annotate_midi

def process_midi(path_in, path_out):
    mid = mido.MidiFile(path_in)

    notes = extract_notes(mid)

    ts = time_signatures(mid.tracks)[0][1]
    beats = ticks_to_beats(notes, mid.ticks_per_beat)
    bars = beats_to_bars(beats, ts)
    phrases = bars_to_phrases(bars)

    bar_chords = harmony_by_bar(
        notes=notes,
        bars=bars,
        ticks_per_beat=mid.ticks_per_beat
    )

    annotated = annotate_midi(
        mid=mid,
        bars=bars,
        phrases=phrases,
        bar_chords=bar_chords,
        key="C:maj",  # заглушка
        ticks_per_beat=mid.ticks_per_beat
    )

    annotated.save(path_out)


if __name__ == "__main__":
    in_dir = Path("data/raw")
    out_dir = Path("data/annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    for midi_file in in_dir.glob("*.mid"):
        print(f"Processing {midi_file.name}")
        process_midi(midi_file, out_dir / midi_file.name)
