"""
main.py — точка входа.

Запуск:
    python main.py

Что делает:
1. Читает все .mid файлы из data/raw/
2. Строит feature vectors для каждого бара
3. Считает SSM (матрицу косинусных сходств)
4. Строит граф inheritance
5. Считает метрики и сохраняет в results/experiment_metrics.csv
6. Сохраняет SSM heatmap для каждого трека в results/ssm/
"""

import sys
from pathlib import Path

import mido
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Добавляем src в путь ---
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from midi.extract import extract_notes, time_signatures
from analysis.structure import ticks_to_beats, beats_to_bars
from analysis.harmony import harmony_by_bar
from analysis.features import build_feature_vector
from analysis.similarity.similarity_matrix import compute_similarity_matrix
from analysis.similarity.graph_inheritance import find_inheritance_edges
from analysis.metrics import evaluate_track
from config import WEIGHTS


# =====================================================
# PROCESS SINGLE MIDI
# =====================================================

def process_midi(path):
    mid = mido.MidiFile(path)
    notes = extract_notes(mid)

    if not notes:
        print(f"  WARNING: no notes found in {path.name}")
        return {}

    ts = time_signatures(mid.tracks)[0][1]
    beats = ticks_to_beats(notes, mid.ticks_per_beat)
    bars = beats_to_bars(beats, ts)

    bar_chords = harmony_by_bar(
        notes=notes,
        bars=bars,
        ticks_per_beat=mid.ticks_per_beat,
    )

    feature_vectors = {}
    for bar_index, bar_beats in bars.items():
        segment_start = bar_beats[0] * mid.ticks_per_beat
        segment_end   = (bar_beats[-1] + 1) * mid.ticks_per_beat
        bar_chord = bar_chords.get(bar_index, "")

        features = build_feature_vector(
            notes=notes,
            segment_start=segment_start,
            segment_end=segment_end,
            bar_chord=bar_chord,
        )
        feature_vectors[bar_index] = features

    return feature_vectors


# =====================================================
# VISUALIZE SSM
# =====================================================

def save_ssm_heatmap(matrix: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", origin="upper")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(title)
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Bar index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =====================================================
# MAIN EXPERIMENT LOOP
# =====================================================

if __name__ == "__main__":

    data_dir  = Path("data/raw")
    ssm_dir   = Path("results/ssm")
    ssm_dir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    results = []

    midi_files = list(data_dir.glob("*.mid")) + list(data_dir.glob("*.midi"))
    if not midi_files:
        print(f"No MIDI files found in {data_dir}. Put .mid files there and re-run.")
        sys.exit(1)

    for midi_file in sorted(midi_files):
        print(f"\nProcessing: {midi_file.name}")

        try:
            feature_vectors = process_midi(midi_file)
        except Exception as e:
            print(f"  ERROR during feature extraction: {e}")
            continue

        if len(feature_vectors) < 2:
            print(f"  SKIP: too few bars ({len(feature_vectors)})")
            continue

        print(f"  Bars: {len(feature_vectors)}")

        # --- SSM ---
        matrices = compute_similarity_matrix(
            feature_vectors=feature_vectors,
            weights=WEIGHTS,
        )

        # Save heatmap
        heatmap_path = ssm_dir / f"{midi_file.stem}_ssm.png"
        save_ssm_heatmap(
            matrix=matrices["total"],
            title=f"SSM: {midi_file.stem}",
            out_path=heatmap_path,
        )
        print(f"  SSM heatmap saved → {heatmap_path}")

        # --- Graph ---
        edges = find_inheritance_edges(
            feature_vectors=feature_vectors,
            weights=WEIGHTS,
        )
        print(f"  Inheritance edges: {len(edges)}")

        # --- Metrics ---
        metrics = evaluate_track(
            similarity_matrix=matrices["total"],
            graph_edges=edges,
        )
        metrics["track"]    = midi_file.stem
        metrics["num_bars"] = len(feature_vectors)

        results.append(metrics)

    if not results:
        print("No results. Check your MIDI files.")
        sys.exit(1)

    # --- Save CSV ---
    df = pd.DataFrame(results)
    csv_path = Path("results/experiment_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Experiment Finished ===")
    print(df.to_string(index=False))
    print(f"\nCSV saved → {csv_path}")