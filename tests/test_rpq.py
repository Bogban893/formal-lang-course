import networkx as nx
import pytest

from project.automata import graph_to_nfa, regex_to_dfa
from project.rpq import AdjacencyMatrixFA, intersect_automata, tensor_based_rpq


class TestAdjacencyMatrixFA:
    def test_adjacency_matrix_fa_builds_matrix_per_symbol(self):
        fa = AdjacencyMatrixFA(regex_to_dfa("a b*"))

        assert fa.num_states == 2
        assert len(fa.start_states) == 1
        assert len(fa.final_states) == 1
        assert {symbol.value for symbol in fa.matrices} == {"a", "b"}
        for matrix in fa.matrices.values():
            assert matrix.shape == (2, 2)

    @pytest.mark.parametrize(
        "regex, accepted, rejected",
        [
            ("a b*", [["a"], ["a", "b"], ["a", "b", "b"]], [[], ["b"], ["a", "a"]]),
            ("a|b", [["a"], ["b"]], [["a", "b"], []]),
            ("(a|b)*", [[], ["a"], ["b", "a", "b"]], [["c"]]),
        ],
    )
    def test_adjacency_matrix_fa_accepts_expected_language(
        self, regex, accepted, rejected
    ):
        fa = AdjacencyMatrixFA(regex_to_dfa(regex))

        for word in accepted:
            assert fa.accepts(word)
        for word in rejected:
            assert not fa.accepts(word)

    def test_adjacency_matrix_fa_from_graph(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="b")

        fa = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {2}))

        assert fa.num_states == 3
        assert fa.accepts(["a", "b"])
        assert not fa.accepts(["a"])

    def test_adjacency_matrix_fa_is_not_empty(self):
        fa = AdjacencyMatrixFA(regex_to_dfa("a b c"))

        assert not fa.is_empty()

    def test_adjacency_matrix_fa_is_empty_when_final_unreachable(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")

        fa = AdjacencyMatrixFA(graph_to_nfa(graph, {1}, {0}))

        assert fa.is_empty()


class TestIntersectAutomata:
    def test_intersect_automata_accepts_common_words(self):
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a*"))
        fa2 = AdjacencyMatrixFA(regex_to_dfa("a a*"))

        intersection = intersect_automata(fa1, fa2)

        assert intersection.num_states == fa1.num_states * fa2.num_states
        assert intersection.accepts(["a"])
        assert intersection.accepts(["a", "a", "a"])
        assert not intersection.accepts([])

    def test_intersect_automata_with_disjoint_languages_is_empty(self):
        fa1 = AdjacencyMatrixFA(regex_to_dfa("a*"))
        fa2 = AdjacencyMatrixFA(regex_to_dfa("b b*"))

        intersection = intersect_automata(fa1, fa2)

        assert intersection.is_empty()


class TestTensorBasedRpq:
    @staticmethod
    def _build_linear_graph() -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="b")
        graph.add_edge(2, 3, label="b")
        return graph

    def test_tensor_based_rpq_linear_graph(self):
        graph = self._build_linear_graph()

        result = tensor_based_rpq("a b*", graph, {0, 1, 2, 3}, {0, 1, 2, 3})

        assert result == {(0, 1), (0, 2), (0, 3)}

    def test_tensor_based_rpq_respects_start_and_final_nodes(self):
        graph = self._build_linear_graph()

        result = tensor_based_rpq("a b*", graph, {0}, {2})

        assert result == {(0, 2)}

    def test_tensor_based_rpq_empty_word_gives_reflexive_pairs(self):
        graph = self._build_linear_graph()

        result = tensor_based_rpq("b*", graph, {1, 2}, {1, 2})

        assert result == {(1, 1), (1, 2), (2, 2)}

    def test_tensor_based_rpq_no_matching_paths(self):
        graph = self._build_linear_graph()

        result = tensor_based_rpq("c", graph, {0, 1, 2, 3}, {0, 1, 2, 3})

        assert result == set()
