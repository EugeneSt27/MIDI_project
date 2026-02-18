# src/config.py
# Параметры для анализа MIDI и построения графа знаний

# Параметры для find_parents (сравнение сегментов)
SIMILARITY_THRESHOLD = 0.75  # Порог схожести для inheritance
MAX_PARENTS = 3  # Максимальное число родителей для сегмента

# Весы для многомерного сходства
WEIGHTS = {
    "rel_pitch": 0.30,
    "rhythm_hist": 0.25,
    "chord_feat": 0.20,
    "abs_pitch": 0.10,
    "density": 0.05,
    "harm_complex": 0.05,
    "note_count": 0.05
}


# Другие параметры
BARS_PER_PHRASE = 4  # Число баров в фразе (для structure.py)
DEFAULT_KEY = "C:maj"  # Заглушка для ключа

# Параметры для визуализации
GRAPH_TITLE_PREFIX = "Knowledge Graph for"  # Префикс для заголовка графа