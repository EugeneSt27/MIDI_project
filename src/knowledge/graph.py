import networkx as nx
import matplotlib.pyplot as plt


class MusicKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    # -------------------------------------------------
    # УЗЛЫ
    # -------------------------------------------------

    def add_phrase(self, phrase_id, features=None):
        """
        phrase_id : int
        features  : dict
        """
        self.graph.add_node(
            f"PHRASE_{phrase_id}",
            type="phrase",
            features=features or {}
        )

    # -------------------------------------------------
    # РЁБРА (ОТНОШЕНИЯ)
    # -------------------------------------------------

    def add_relation(self, src, dst, relation, weight=None):
        """
        relation: repeat | variation | contrast | development
        """
        self.graph.add_edge(
            f"PHRASE_{src}",
            f"PHRASE_{dst}",
            relation=relation,
            weight=weight
        )

    # -------------------------------------------------
    # ПОСТРОЕНИЕ ИЗ АНАЛИЗА
    # -------------------------------------------------

    def build_from_analysis(self, phrases, relations, features):
        """
        phrases   : iterable of phrase ids
        relations : [(src, dst, relation_type, score)]
        features  : {phrase_id: feature_vector}
        """

        for p in phrases:
            self.add_phrase(p, features.get(p))

        for src, dst, rel, score in relations:
            self.add_relation(src, dst, rel, score)

    # -------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ
    # -------------------------------------------------

    def draw(self):
        pos = nx.spring_layout(self.graph, seed=42)

        edge_labels = {
            (u, v): d["relation"]
            for u, v, d in self.graph.edges(data=True)
        }

        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=2500,
            node_color="lightblue",
            font_size=9
        )

        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=8
        )

        plt.title("Music Knowledge Graph")
        plt.show()

    # -------------------------------------------------
    # ДЛЯ БУДУЩЕЙ ГЕНЕРАЦИИ
    # -------------------------------------------------

    def get_successors(self, phrase_id):
        return list(self.graph.successors(f"PHRASE_{phrase_id}"))

    def get_relation(self, src, dst):
        return self.graph.edges[
            f"PHRASE_{src}",
            f"PHRASE_{dst}"
        ]

