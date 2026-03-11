"""
phrase_analysis.py — позиционное сравнение тактов между фразами.

Реализует идею научника: сравниваем не фразы целиком,
а такт №k одной фразы с тактом №k другой фразы — по каждому
компоненту (harmony, melody, rhythm) отдельно.

Результат — таблица паттернов изменений, как в примере с Nefeli:

    a1 - e2 - a3 - a4
    a1 - e5 - a3 - a4   ← мелодия поменялась в позиции 1
    F1 - e2 - a3 - a4   ← гармония поменялась в позиции 0
    F6 - e7 - a3 - a4

И рекурсивное сравнение: фразы → секции → всё произведение.
"""

import numpy as np
from typing import Dict, List, Tuple
from analysis.features import build_bar_profile
from analysis.similarity.base import cosine_similarity


# порог для классификации сходства на HIGH/LOW
SIMILARITY_THRESHOLD = 0.75


# -------------------------------------------------------
# POSITIONAL SIMILARITY
# -------------------------------------------------------

def positional_similarity(
    phrases: List[List[int]],
    bar_profiles: Dict[int, Dict[str, np.ndarray]],
) -> Dict:
    """
    Для каждой позиции внутри фразы (0, 1, 2, 3, ...)
    сравнивает этот такт между всеми парами фраз.

    Возвращает:
    {
        position_0: {
            "harmony": np.ndarray (num_phrases x num_phrases),
            "melody":  np.ndarray,
            "rhythm":  np.ndarray,
        },
        position_1: { ... },
        ...
    }

    Каждая матрица [i, j] = сходство такта на позиции k
                             между фразой i и фразой j.
    """
    if not phrases:
        return {}

    # максимальная длина фразы
    max_len = max(len(p) for p in phrases)
    num_phrases = len(phrases)
    components = ["harmony", "melody", "rhythm"]

    result = {}

    for pos in range(max_len):
        matrices = {c: np.zeros((num_phrases, num_phrases)) for c in components}

        for i, phrase_i in enumerate(phrases):
            for j, phrase_j in enumerate(phrases):
                if pos >= len(phrase_i) or pos >= len(phrase_j):
                    # позиция не существует в одной из фраз
                    for c in components:
                        matrices[c][i, j] = np.nan
                    continue

                bar_i = phrase_i[pos]
                bar_j = phrase_j[pos]

                profile_i = bar_profiles[bar_i]
                profile_j = bar_profiles[bar_j]

                for c in components:
                    matrices[c][i, j] = cosine_similarity(
                        profile_i[c], profile_j[c]
                    )

        result[pos] = matrices

    return result


# -------------------------------------------------------
# CHANGE PATTERNS
# -------------------------------------------------------

def classify_similarity(value: float, threshold: float = SIMILARITY_THRESHOLD) -> str:
    """Классифицирует сходство как H (high) или L (low)."""
    if np.isnan(value):
        return "?"
    return "H" if value >= threshold else "L"


def extract_change_patterns(
    phrases: List[List[int]],
    bar_profiles: Dict[int, Dict[str, np.ndarray]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """
    Для каждой пары соседних фраз (i, i+1) извлекает паттерн изменений
    по каждой позиции и каждому компоненту.

    Возвращает список паттернов:
    [
        {
            "phrase_pair": (0, 1),
            "patterns": {
                0: {"harmony": "H", "melody": "L", "rhythm": "H"},  # позиция 0
                1: {"harmony": "H", "melody": "H", "rhythm": "H"},  # позиция 1
                ...
            },
            "change_type": "remelodization" | "reharmonization" | "full_change" | "repetition" | "mixed"
        },
        ...
    ]
    """
    pos_sims = positional_similarity(phrases, bar_profiles)
    components = ["harmony", "melody", "rhythm"]
    result = []

    for i in range(len(phrases) - 1):
        patterns = {}

        for pos, matrices in pos_sims.items():
            if pos >= len(phrases[i]) or pos >= len(phrases[i+1]):
                continue

            pattern = {
                c: classify_similarity(matrices[c][i, i+1], threshold)
                for c in components
            }
            patterns[pos] = pattern

        change_type = _classify_change_type(patterns)

        result.append({
            "phrase_pair": (i, i + 1),
            "patterns":    patterns,
            "change_type": change_type,
        })

    return result


def _classify_change_type(patterns: Dict[int, Dict[str, str]]) -> str:
    """
    Определяет тип изменения между двумя фразами на основе паттернов.

    Типы (по идее научника):
        repetition      — всё одинаково (H H H везде)
        remelodization  — гармония стабильна, мелодия меняется (H L *)
        reharmonization — мелодия стабильна, гармония меняется (L H *)
        full_change     — всё меняется (L L *)
        mixed           — нет доминирующего паттерна
    """
    if not patterns:
        return "unknown"

    harmony_vals  = [p["harmony"] for p in patterns.values()]
    melody_vals   = [p["melody"]  for p in patterns.values()]

    h_stable = harmony_vals.count("H") > len(harmony_vals) / 2
    m_stable = melody_vals.count("H")  > len(melody_vals)  / 2

    if h_stable and m_stable:
        return "repetition"
    elif h_stable and not m_stable:
        return "remelodization"
    elif not h_stable and m_stable:
        return "reharmonization"
    elif not h_stable and not m_stable:
        return "full_change"
    else:
        return "mixed"


# -------------------------------------------------------
# RECURSIVE ANALYSIS
# -------------------------------------------------------

def recursive_structure_analysis(
    sections: List[List[int]],
    bar_profiles: Dict[int, Dict[str, np.ndarray]],
    bars_per_phrase: int = 4,
    threshold: float = SIMILARITY_THRESHOLD,
) -> Dict:
    """
    Рекурсивный анализ структуры на двух уровнях:

    Уровень 1 (бары внутри фраз):
        Сравниваем такты позиционно между соседними фразами.
        → паттерны типа remelodization / reharmonization

    Уровень 2 (фразы внутри секций):
        Каждую фразу представляем как вектор (среднее профилей её баров),
        сравниваем фразы между секциями.
        → паттерны повторения секций (ABAB, AABA и т.д.)

    Возвращает:
    {
        "level1": {
            section_idx: [change_patterns...]
        },
        "level2": {
            "phrase_labels": ["A", "A", "B", "A", ...],  # авто-разметка
            "section_pattern": "AABA",
        }
    }
    """
    from analysis.novelty import split_into_phrases

    result = {"level1": {}, "level2": {}}

    # --- LEVEL 1: бары внутри фраз ---
    all_phrases_flat = []
    for sec_idx, section in enumerate(sections):
        phrases = split_into_phrases(section, bars_per_phrase)
        if len(phrases) < 2:
            continue
        patterns = extract_change_patterns(phrases, bar_profiles, threshold)
        result["level1"][sec_idx] = patterns
        all_phrases_flat.extend(phrases)

    # --- LEVEL 2: фразы → буквенная разметка ---
    if len(all_phrases_flat) >= 2:
        phrase_vectors = _phrase_to_vector(all_phrases_flat, bar_profiles)
        labels, pattern_str = _label_phrases(phrase_vectors, threshold)
        result["level2"]["phrase_labels"]   = labels
        result["level2"]["section_pattern"] = pattern_str

    return result


def _phrase_to_vector(
    phrases: List[List[int]],
    bar_profiles: Dict[int, Dict[str, np.ndarray]],
) -> List[np.ndarray]:
    """
    Представляет каждую фразу как среднее профилей её баров (плоский вектор).
    """
    vectors = []
    for phrase in phrases:
        bars_in_phrase = [bar_profiles[b] for b in phrase if b in bar_profiles]
        if not bars_in_phrase:
            vectors.append(np.zeros(48))
            continue

        # конкатенируем компоненты каждого бара и усредняем
        flat_bars = []
        for profile in bars_in_phrase:
            flat = np.concatenate([profile["harmony"], profile["melody"], profile["rhythm"]])
            flat_bars.append(flat)

        vectors.append(np.mean(flat_bars, axis=0))

    return vectors


def _label_phrases(
    phrase_vectors: List[np.ndarray],
    threshold: float,
) -> Tuple[List[str], str]:
    """
    Автоматически присваивает буквенные метки фразам (A, B, C, ...).

    Алгоритм: первая фраза = A. Каждая следующая сравнивается со всеми
    предыдущими прототипами. Если сходство выше порога — та же буква,
    иначе — новая буква.

    Возвращает (labels, pattern_string), например (["A","A","B","A"], "AABA").
    """
    if not phrase_vectors:
        return [], ""

    labels = []
    prototypes = []  # (буква, вектор)
    letter_idx = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for vec in phrase_vectors:
        matched = None
        best_sim = -1

        for letter, proto in prototypes:
            sim = cosine_similarity(vec, proto)
            if sim > best_sim:
                best_sim = sim
                matched = letter

        if best_sim >= threshold:
            labels.append(matched)
        else:
            new_letter = alphabet[letter_idx % len(alphabet)]
            letter_idx += 1
            labels.append(new_letter)
            prototypes.append((new_letter, vec))

    pattern_str = "".join(labels)
    return labels, pattern_str
    
# -------------------------------------------------------
# SUMMARY REPORT
# -------------------------------------------------------

def print_structure_report(analysis: Dict, track_name: str = ""):
    """Выводит читаемый отчёт об анализе структуры."""
    print(f"\n{'='*50}")
    print(f"STRUCTURE REPORT: {track_name}")
    print(f"{'='*50}")

    # Level 2: буквенная разметка
    l2 = analysis.get("level2", {})
    if "phrase_labels" in l2:
        print(f"\nPhrase pattern: {l2['section_pattern']}")
        print(f"Labels: {l2['phrase_labels']}")

    # Level 1: паттерны изменений
    l1 = analysis.get("level1", {})
    if l1:
        print(f"\nChange patterns between consecutive phrases:")
        for sec_idx, patterns in l1.items():
            print(f"  Section {sec_idx}:")
            for p in patterns:
                i, j = p["phrase_pair"]
                ct = p["change_type"]
                pos_summary = []
                for pos, pat in p["patterns"].items():
                    pos_summary.append(
                        f"pos{pos}:[H={pat['harmony']},M={pat['melody']},R={pat['rhythm']}]"
                    )
                print(f"    phrase {i}→{j}: {ct:20s}  {' '.join(pos_summary)}")