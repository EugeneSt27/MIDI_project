# src/config.py
# Параметры для анализа MIDI и построения графа знаний

# Параметры для find_parents (сравнение сегментов)
SIMILARITY_THRESHOLD = 0.75  # Порог схожести для inheritance
MAX_PARENTS = 3  # Максимальное число родителей для сегмента

# Весы для многомерного сходства (pitch, chord, rhythm, struct)
WEIGHTS = {
    'pitch': 1.0,    # Вес для pitch features
    'chord': 1.5,    # Вес для harmony features
    'rhythm': 2.0,   # Вес для rhythm features
    'struct': 1.0    # Вес для structural features
}

# Другие параметры
BARS_PER_PHRASE = 4  # Число баров в фразе (для structure.py)
DEFAULT_KEY = "C:maj"  # Заглушка для ключа

# Параметры для визуализации
GRAPH_TITLE_PREFIX = "Knowledge Graph for"  # Префикс для заголовка графа