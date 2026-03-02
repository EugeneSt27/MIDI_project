"""
main.py — точка входа.
Pipeline: MIDI → bar profiles → SSM (total + per component) →
          novelty detection → phrase analysis → metrics → CSV
"""

import sys
from pathlib import Path
import mido
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from midi.extract import extract_notes, time_signatures
from analysis.structure import ticks_to_beats, beats_to_bars
from analysis.harmony import harmony_by_bar
from analysis.features import build_bar_profile, build_feature_vector
from analysis.similarity.base import cosine_similarity
from analysis.similarity.similarity_matrix import compute_similarity_matrix
from analysis.similarity.graph_inheritance import find_inheritance_edges
from analysis.metrics import evaluate_track
from analysis.novelty import analyze_structure
from analysis.phrase_analysis import recursive_structure_analysis, print_structure_report
from config import WEIGHTS


def process_midi(path):
    mid = mido.MidiFile(path)
    notes = extract_notes(mid)
    if not notes:
        return {}, {}
    ts = time_signatures(mid.tracks)[0][1]
    beats = ticks_to_beats(notes, mid.ticks_per_beat)
    bars = beats_to_bars(beats, ts)
    bar_chords = harmony_by_bar(notes=notes, bars=bars, ticks_per_beat=mid.ticks_per_beat)

    bar_profiles, feature_vectors = {}, {}
    for bar_index, bar_beats in bars.items():
        s = bar_beats[0] * mid.ticks_per_beat
        e = (bar_beats[-1] + 1) * mid.ticks_per_beat
        chord = bar_chords.get(bar_index, "")
        bar_profiles[bar_index] = build_bar_profile(notes, s, e, chord)
        feature_vectors[bar_index] = build_feature_vector(notes, s, e, chord)
    return bar_profiles, feature_vectors


def save_ssm_heatmap(matrix, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", origin="upper")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(title); ax.set_xlabel("Bar"); ax.set_ylabel("Bar")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def save_novelty_plot(novelty_curve, boundaries, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(novelty_curve, color="#2196F3", linewidth=1.5, label="Novelty")
    for b in boundaries[1:-1]:
        ax.axvline(x=b, color="#FF5722", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_title(f"Novelty: {title}"); ax.set_xlabel("Bar"); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def save_component_ssm(component_matrices, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, comp in zip(axes, ["harmony", "melody", "rhythm"]):
        im = ax.imshow(component_matrices[comp], vmin=0, vmax=1, cmap="magma", origin="upper")
        plt.colorbar(im, ax=ax); ax.set_title(f"{comp.capitalize()} SSM")
        ax.set_xlabel("Bar"); ax.set_ylabel("Bar")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


if __name__ == "__main__":
    data_dir = Path("data/raw")
    Path("results/ssm").mkdir(parents=True, exist_ok=True)
    Path("results/novelty").mkdir(parents=True, exist_ok=True)

    midi_files = sorted(list(data_dir.glob("*.mid")) + list(data_dir.glob("*.midi")))
    if not midi_files:
        print(f"No MIDI files in {data_dir}"); sys.exit(1)

    results = []
    for midi_file in midi_files:
        print(f"\n{'─'*50}\nProcessing: {midi_file.name}")
        try:
            bar_profiles, feature_vectors = process_midi(midi_file)
        except Exception as e:
            print(f"  ERROR: {e}"); continue

        if len(feature_vectors) < 4:
            print(f"  SKIP: too few bars"); continue

        bar_ids = sorted(feature_vectors.keys())
        print(f"  Bars: {len(bar_ids)}")

        # SSM total
        matrices = compute_similarity_matrix(feature_vectors=feature_vectors, weights=WEIGHTS)
        save_ssm_heatmap(matrices["total"], f"SSM: {midi_file.stem}",
                         f"results/ssm/{midi_file.stem}_ssm_total.png")

        # SSM per component
        n = len(bar_ids)
        comp_mat = {}
        for comp in ["harmony", "melody", "rhythm"]:
            mat = np.zeros((n, n))
            for i, bi in enumerate(bar_ids):
                for j, bj in enumerate(bar_ids):
                    mat[i, j] = cosine_similarity(bar_profiles[bi][comp], bar_profiles[bj][comp])
            comp_mat[comp] = mat
        save_component_ssm(comp_mat, f"Component SSMs: {midi_file.stem}",
                           f"results/ssm/{midi_file.stem}_ssm_components.png")

        # Novelty + structure
        structure = analyze_structure(sim_matrix=matrices["total"], bar_ids=bar_ids,
                                      kernel_size=4, threshold=0.3, min_section_bars=4, bars_per_phrase=4)
        save_novelty_plot(structure["novelty_curve"], structure["boundaries"],
                          midi_file.stem, f"results/novelty/{midi_file.stem}_novelty.png")
        print(f"  Sections: {structure['num_sections']}, Phrases: {structure['num_phrases']}")

        # Phrase analysis
        phrase_analysis = recursive_structure_analysis(
            sections=structure["sections"], bar_profiles=bar_profiles,
            bars_per_phrase=4, threshold=0.75)
        print_structure_report(phrase_analysis, midi_file.stem)

        # Graph + metrics
        edges = find_inheritance_edges(feature_vectors=feature_vectors, weights=WEIGHTS)
        metrics = evaluate_track(similarity_matrix=matrices["total"], graph_edges=edges)
        metrics.update({
            "track": midi_file.stem, "num_bars": len(bar_ids),
            "num_sections": structure["num_sections"], "num_phrases": structure["num_phrases"],
            "phrase_pattern": phrase_analysis.get("level2", {}).get("section_pattern", ""),
        })
        results.append(metrics)

    if not results:
        print("No results."); sys.exit(1)

    df = pd.DataFrame(results)
    df.to_csv("results/experiment_metrics.csv", index=False)
    print(f"\n{'='*50}\nEXPERIMENT FINISHED")
    cols = ["track", "num_bars", "num_sections", "phrase_pattern", "block_score", "structure_score"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))