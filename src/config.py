# src/config.py
# Параметры для анализа MIDI и построения графа знаний


WEIGHTS = {"harmony": 1.0, "melody": 1.0, "rhythm": 1.0}
SIMILARITY_THRESHOLD = 0.75
MAX_PARENTS = 3
BARS_PER_PHRASE = 4


# Другие параметры
BARS_PER_PHRASE = 4  # Число баров в фразе (для structure.py)
DEFAULT_KEY = "C:maj"  # Заглушка для ключа

# Параметры для визуализации
GRAPH_TITLE_PREFIX = "Knowledge Graph for"  # Префикс для заголовка графа