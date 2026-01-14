import mido
from pathlib import Path
import mido
import pickle

from src.midi.extract import extract_notes, time_signatures
from src.analysis.structure import (
    ticks_to_beats,
    beats_to_bars,
    bars_to_phrases
)
from src.analysis.harmony import harmony_by_bar
from src.midi.annotate import annotate_midi
from src.analysis.features import build_feature_vector
from src.analysis.development import find_parents
from src.knowledge.graph import build_knowledge_graph, analyze_graph, draw_graph, draw_beautiful_graph
from src.config import SIMILARITY_THRESHOLD, MAX_PARENTS, WEIGHTS, DEFAULT_KEY, GRAPH_TITLE_PREFIX

def process_midi(path_in, path_out, midi_name):
    """
    Выполняет разметку MIDI, деление на блоки (фразы) и их сравнение.
    Возвращает phrases, features, edges для дальнейшего анализа.
    """
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
        key=DEFAULT_KEY,  # из config
        ticks_per_beat=mid.ticks_per_beat
    )
    
    annotated.save(path_out)
    
    # Собираем features для фраз
    file_phrases = {}
    file_features = {}
    for p_idx, phrase in phrases.items():
        if not phrase:
            continue
        # Границы фразы
        segment_start = bars[phrase[0]][0] * mid.ticks_per_beat
        segment_end = (bars[phrase[-1]][-1] + 1) * mid.ticks_per_beat
        # Chord для фразы (первый бар)
        bar_chord = bar_chords.get(phrase[0], "")
        # Bar index для структуры (первый бар фразы)
        bar_index = phrase[0] + 1
        phrase_id = p_idx + 1
        
        features = build_feature_vector(
            notes=notes,
            segment_start=segment_start,
            segment_end=segment_end,
            bar_chord=bar_chord,
            bar_index=bar_index,
            phrase_index=phrase_id
        )
        
        file_phrases[phrase_id] = phrase
        file_features[phrase_id] = features
        
        print(f"Features for phrase {phrase_id} in {midi_name}: {features[:5]}...")  # первые 5 элементов
    
    # Сравнение: находим edges (inheritance)
    edges = find_parents(
        feature_vectors=file_features,
        similarity_threshold=SIMILARITY_THRESHOLD,
        max_parents=MAX_PARENTS,
        weights=WEIGHTS
    )
    
    return file_phrases, file_features, edges

def analyze_and_visualize(phrases, features, midi_name):
    """
    Строит граф, анализирует и визуализирует результаты.
    """
    # Строим граф
    G = build_knowledge_graph(
        segments=phrases,
        features=features,
        weights=WEIGHTS
    )
    
    # Анализируем
    stats = analyze_graph(G)
    print(f"Graph Analysis for {midi_name}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Сохраняем данные и граф
    data_path = Path("results/data") / f"{midi_name}_data.pkl"
    graph_path = Path("results/graphs") / f"{midi_name}_graph.png"
    
    with open(data_path, 'wb') as f:
        pickle.dump((phrases, features), f)  # сохраняем phrases и features
    print(f"Data saved to {data_path}")
    
    draw_graph(G, title=f"{GRAPH_TITLE_PREFIX} {midi_name}", save_path=str(graph_path))
    print(f"Graph saved to {graph_path}")
    
    # Beautiful interactive graph
    beautiful_graph_path = Path("results/better_graphs") / f"{midi_name}_beautiful_graph.html"
    draw_beautiful_graph(G, title=f"Beautiful {GRAPH_TITLE_PREFIX} {midi_name}", save_path=str(beautiful_graph_path))
    print(f"Beautiful graph saved to {beautiful_graph_path}")

if __name__ == "__main__":
    in_dir = Path("data/raw")
    out_dir = Path("data/annotated")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for midi_file in in_dir.glob("*.mid"):
        midi_name = midi_file.stem  # название без расширения
        print(f"Processing {midi_name}")
        
        # Разметка, деление и сравнение
        phrases, features, _ = process_midi(midi_file, out_dir / midi_file.name, midi_name)
        
        # Анализ и визуализация
        analyze_and_visualize(phrases, features, midi_name)
