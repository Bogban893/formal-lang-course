import pytest
import networkx as nx

from pyformlang.cfg import CFG
from project.gll_cfpq import gll_based_cfpq
from project.tensor_cfpq import cfg_to_rsm


class TestGll:
    def test_only_complete_paths_are_returned(self):
        graph = nx.DiGraph()
        graph.add_edge(10, 20, label="a")
        graph.add_edge(20, 30, label="b")
        cfg = CFG.from_text("S -> a b")

        assert gll_based_cfpq(cfg_to_rsm(cfg), graph) == {(10, 30)}

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

        assert gll_based_cfpq(cfg_to_rsm(cfg), graph, start_nodes, final_nodes) == expected

    def test_epsilon_on_isolated_vertices(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([10, 20])
        cfg = CFG.from_text("S -> epsilon")

        assert gll_based_cfpq(cfg_to_rsm(cfg), graph) == {(10, 10), (20, 20)}

    def test_recursive_paths(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="a")
        graph.add_edge(2, 3, label="a")
        cfg = CFG.from_text("S -> a | S S")

        assert gll_based_cfpq(cfg_to_rsm(cfg), graph) == {
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

        assert gll_based_cfpq(cfg_to_rsm(cfg), graph) == {(0, 1)}

    def test_empty_graph(self):
        cfg = CFG.from_text("S -> a | epsilon")

        assert gll_based_cfpq(cfg_to_rsm(cfg), nx.DiGraph()) == set()
