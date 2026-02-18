import sys
from pathlib import Path
import mido
import pandas as pd

# --- Добавляем src в путь ---
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from midi.extract import extract_notes, time_signatures
from analysis.structure import ticks_to_beats, beats_to_bars
from analysis.harmony import harmony_by_bar
from analysis.features import build_feature_vector

from analysis.similarity.similarity_matrix import compute_similarity_matrix
from analysis.similarity.clustering import cluster_bars_dbscan
from analysis.similarity.graph_inheritance import find_inheritance_edges

from analysis.metrics import evaluate_track

from config import WEIGHTS


# =====================================================
# PROCESS SINGLE MIDI
# =====================================================

def process_midi(path):
    mid = mido.MidiFile(path)
    notes = extract_notes(mid)
    ts = time_signatures(mid.tracks)[0][1]
    beats = ticks_to_beats(notes, mid.ticks_per_beat)
    bars = beats_to_bars(beats, ts)

    bar_chords = harmony_by_bar(
        notes=notes,
        bars=bars,
        ticks_per_beat=mid.ticks_per_beat
    )

    feature_vectors = {}

    for bar_index, bar in bars.items():
        segment_start = bar[0] * mid.ticks_per_beat
        segment_end = (bar[-1] + 1) * mid.ticks_per_beat

        bar_chord = bar_chords.get(bar_index, "")

        features = build_feature_vector(
            notes=notes,
            segment_start=segment_start,
            segment_end=segment_end,
            bar_chord=bar_chord
        )

        feature_vectors[bar_index] = features
    #print(feature_vectors[1])
    return feature_vectors


# =====================================================
# MAIN EXPERIMENT LOOP
# =====================================================

if __name__ == "__main__":

    data_dir = Path("data/raw")
    results = []

    for midi_file in data_dir.glob("*.mid"):

        print(f"Processing {midi_file.name}")

        feature_vectors = process_midi(midi_file)

        # --- SSM ---
        matrices = compute_similarity_matrix(
            feature_vectors=feature_vectors,
            weights=WEIGHTS
        )

        # --- Clustering ---
       # labels = cluster_bars_dbscan(
       #     feature_vectors=feature_vectors,
       #     weights=WEIGHTS
       # )

        # --- Graph ---
        edges = find_inheritance_edges(
            feature_vectors=feature_vectors,
            weights=WEIGHTS
        )

        # --- Metrics ---
        metrics = evaluate_track(
            similarity_matrix=matrices["total"],
           # clustering_labels=labels,
            graph_edges=edges
        )

        metrics["track"] = midi_file.stem
        metrics["num_bars"] = len(feature_vectors)

        results.append(metrics)

    # --- Save to CSV ---
    df = pd.DataFrame(results)
    df.to_csv("results/experiment_metrics.csv", index=False)

    print("\n=== Experiment Finished ===")
    print(df)
