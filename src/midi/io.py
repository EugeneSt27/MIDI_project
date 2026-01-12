import mido
from pathlib import Path

def load_midi(path):
    return mido.MidiFile(path)

def save_midi(mid, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(path)
