import pytest
import networkx as nx

from pyformlang.cfg import CFG
from project.matrix_cfpq import matrix_based_cfpq


class TestHellings:
    def test_only_complete_paths_are_returned(self):
        graph = nx.DiGraph()
        graph.add_edge(10, 20, label="a")
        graph.add_edge(20, 30, label="b")
        cfg = CFG.from_text("S -> a b")

        assert matrix_based_cfpq(cfg, graph) == {(10, 30)}

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

        assert matrix_based_cfpq(cfg, graph, start_nodes, final_nodes) == expected

    def test_epsilon_on_isolated_vertices(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([10, 20])
        cfg = CFG.from_text("S -> epsilon")

        assert matrix_based_cfpq(cfg, graph) == {(10, 10), (20, 20)}

    def test_recursive_paths(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="a")
        graph.add_edge(2, 3, label="a")
        cfg = CFG.from_text("S -> a | S S")

        assert matrix_based_cfpq(cfg, graph) == {
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

        assert matrix_based_cfpq(cfg, graph) == {(0, 1)}

    def test_empty_graph(self):
        cfg = CFG.from_text("S -> a | epsilon")

        assert matrix_based_cfpq(cfg, nx.DiGraph()) == set()
