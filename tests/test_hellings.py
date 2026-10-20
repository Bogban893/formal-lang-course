import pytest
import networkx as nx

from pyformlang.cfg import CFG, Variable, Terminal
from project.hellings import cfg_to_weak_normal_form, hellings_based_cfpq


class TestWeakNormalForm:
    @pytest.mark.parametrize(
        "grammar",
        [
            "S -> a",
            "S -> epsilon",
            "S -> A B\nA -> a\nB -> b",
            "S -> a S b | epsilon",
            "S -> A\nA -> B\nB -> a",
            "S -> a b c d",
        ],
    )
    def test_result_is_weak_normal_form(self, grammar):
        cfg = CFG.from_text(grammar)

        result = cfg_to_weak_normal_form(cfg)

        for production in result.productions:
            body = production.body

            assert (
                len(body) == 0
                or (len(body) == 1 and isinstance(body[0], Terminal))
                or (len(body) == 2 and all(isinstance(x, Variable) for x in body))
            )

    @pytest.mark.parametrize(
        "grammar, words",
        [
            ("S -> a", [[], ["a"], ["b"]]),
            ("S -> epsilon", [[], ["a"]]),
            ("S -> a b", [["a", "b"], ["a"], ["b", "a"]]),
            (
                "S -> a S b | epsilon",
                [[], ["a", "b"], ["a", "a", "b", "b"], ["a", "b", "b"]],
            ),
            ("S -> A\nA -> B\nB -> a", [[], ["a"], ["b"]]),
            ("S -> a b c d", [["a", "b", "c", "d"], ["a", "b"], []]),
        ],
    )
    def test_language_is_preserved(self, grammar, words):
        cfg = CFG.from_text(grammar)

        result = cfg_to_weak_normal_form(cfg)

        for word in words:
            assert cfg.contains(word) == result.contains(word)

    def test_epsilon_is_preserved(self):
        cfg = CFG.from_text("""
        S -> a S b | epsilon
        """)

        result = cfg_to_weak_normal_form(cfg)

        assert result.contains([])

    def test_unit_productions_are_removed(self):
        cfg = CFG.from_text("""
        S -> A
        A -> B
        B -> a
        """)

        result = cfg_to_weak_normal_form(cfg)

        assert result.contains(["a"])

        for production in result.productions:
            assert not (
                len(production.body) == 1 and isinstance(production.body[0], Variable)
            )

    def test_original_grammar_is_not_modified(self):
        cfg = CFG.from_text("""
        S -> a S b | epsilon
        """)

        original_rules = set(cfg.productions)

        cfg_to_weak_normal_form(cfg)

        assert cfg.productions == original_rules

    def test_result_is_cfg(self):
        cfg = CFG.from_text("S -> a b")

        result = cfg_to_weak_normal_form(cfg)

        assert isinstance(result, CFG)
        assert result.start_symbol == cfg.start_symbol


class TestHellings:
    def test_only_complete_paths_are_returned(self):
        graph = nx.DiGraph()
        graph.add_edge(10, 20, label="a")
        graph.add_edge(20, 30, label="b")
        cfg = CFG.from_text("S -> a b")

        assert hellings_based_cfpq(cfg, graph) == {(10, 30)}

    @pytest.mark.parametrize(
        "start_nodes, final_nodes, expected",
        [
            (None, None, {(0, 2), (2, 4)}),
            ({0}, None, {(0, 2)}),
            (None, {4}, {(2, 4)}),
            ({0}, {2}, {(0, 2)}),
            (set(), None, set()),
            (None, set(), set()),
        ],
    )
    def test_vertex_filters(self, start_nodes, final_nodes, expected):
        graph = nx.DiGraph()
        for start, end, label in [(0, 1, "a"), (1, 2, "b"), (2, 3, "a"), (3, 4, "b")]:
            graph.add_edge(start, end, label=label)
        cfg = CFG.from_text("S -> a b")

        assert hellings_based_cfpq(cfg, graph, start_nodes, final_nodes) == expected

    def test_epsilon_on_isolated_vertices(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([10, 20])
        cfg = CFG.from_text("S -> epsilon")

        assert hellings_based_cfpq(cfg, graph) == {(10, 10), (20, 20)}

    def test_recursive_paths(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="a")
        graph.add_edge(2, 3, label="a")
        cfg = CFG.from_text("S -> a | S S")

        assert hellings_based_cfpq(cfg, graph) == {
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 2),
            (1, 3),
            (0, 3),
        }

    def test_parallel_edges(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="b")
        graph.add_edge(0, 1, label="a")
        cfg = CFG.from_text("S -> a")

        assert hellings_based_cfpq(cfg, graph) == {(0, 1)}

    def test_empty_graph(self):
        cfg = CFG.from_text("S -> a | epsilon")

        assert hellings_based_cfpq(cfg, nx.DiGraph()) == set()
