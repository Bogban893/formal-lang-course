import networkx as nx
import pytest

from project.ms_bfs_rpq import ms_bfs_based_rpq
from project.rpq import tensor_based_rpq


class TestMsBfsBasedRpq:
    @staticmethod
    def _build_linear_graph() -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="b")
        graph.add_edge(2, 3, label="b")
        return graph

    @staticmethod
    def _build_cycle_graph() -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="a")
        graph.add_edge(2, 0, label="b")
        return graph

    def test_ms_bfs_based_rpq_linear_graph(self):
        graph = self._build_linear_graph()

        result = ms_bfs_based_rpq("a b*", graph, {0, 1, 2, 3}, {0, 1, 2, 3})

        assert result == {(0, 1), (0, 2), (0, 3)}

    def test_ms_bfs_based_rpq_respects_start_and_final_nodes(self):
        graph = self._build_linear_graph()

        result = ms_bfs_based_rpq("a b*", graph, {0}, {2})

        assert result == {(0, 2)}

    def test_ms_bfs_based_rpq_multiple_start_nodes(self):
        graph = self._build_linear_graph()

        result = ms_bfs_based_rpq("b", graph, {1, 2}, {0, 1, 2, 3})

        assert result == {(1, 2), (2, 3)}

    def test_ms_bfs_based_rpq_empty_word_gives_reflexive_pairs(self):
        graph = self._build_linear_graph()

        result = ms_bfs_based_rpq("b*", graph, {1, 2}, {1, 2})

        assert result == {(1, 1), (1, 2), (2, 2)}

    def test_ms_bfs_based_rpq_no_matching_paths(self):
        graph = self._build_linear_graph()

        result = ms_bfs_based_rpq("c", graph, {0, 1, 2, 3}, {0, 1, 2, 3})

        assert result == set()

    def test_ms_bfs_based_rpq_cycle_graph(self):
        graph = self._build_cycle_graph()

        result = ms_bfs_based_rpq("(a a b)*", graph, {0}, {0, 1, 2})

        assert result == {(0, 0)}

    def test_ms_bfs_based_rpq_default_start_and_final_nodes(self):
        graph = self._build_linear_graph()

        result = ms_bfs_based_rpq("a", graph, set(), set())

        assert result == {(0, 1)}

    @pytest.mark.parametrize("regex", ["a*", "a b", "(a|b)*", "a a* b", "b a*"])
    def test_ms_bfs_based_rpq_matches_tensor_based_rpq(self, regex):
        graph = self._build_cycle_graph()
        start_nodes, final_nodes = {0, 2}, {0, 1}

        ms_bfs = ms_bfs_based_rpq(regex, graph, start_nodes, final_nodes)
        tensor = tensor_based_rpq(regex, graph, start_nodes, final_nodes)

        assert ms_bfs == tensor
