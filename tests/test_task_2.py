import networkx as nx
import pytest
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)

from project.task_2 import regex_to_dfa, graph_to_nfa


class TestRegexToDfa:
    @pytest.mark.parametrize(
        "regex, accepted, rejected",
        [
            ("a*b", [["b"], ["a", "b"], ["a", "a", "b"]], [["a"], [], ["b", "a"]]),
            ("a|b", [["a"], ["b"]], [["a", "b"], []]),
            ("(a|b)*", [[], ["a"], ["b"], ["a", "b", "a"]], [["c"]]),
            ("a.b.c", [["a", "b", "c"]], [["a", "b"], ["a", "c", "b"]]),
        ],
    )
    def test_regex_to_dfa_accepts_expected_language(self, regex, accepted, rejected):
        dfa = regex_to_dfa(regex)

        assert isinstance(dfa, DeterministicFiniteAutomaton)
        for word in accepted:
            assert dfa.accepts(word)
        for word in rejected:
            assert not dfa.accepts(word)

    def test_regex_to_dfa_is_deterministic(self):
        dfa = regex_to_dfa("a*b*c*")

        assert dfa.is_deterministic()

    def test_regex_to_dfa_is_minimal(self):
        dfa = regex_to_dfa("a*b")
        minimal_dfa = dfa.minimize()

        assert len(dfa.states) == len(minimal_dfa.states)

    def test_regex_to_dfa_equivalent_regexes_produce_equal_languages(self):
        dfa_1 = regex_to_dfa("a.b.c*")
        dfa_2 = regex_to_dfa("a.b.(c*)")

        assert dfa_1.is_equivalent_to(dfa_2)

    def test_regex_to_dfa_epsilon_language(self):
        dfa = regex_to_dfa("$")

        assert dfa.accepts([])
        assert not dfa.accepts(["a"])

    @pytest.mark.parametrize("invalid_regex", ["(a|b", "*", "a**b)"])
    def test_regex_to_dfa_raises_on_invalid_regex(self, invalid_regex):
        with pytest.raises(Exception):
            regex_to_dfa(invalid_regex)


class TestGraphToNfa:
    @staticmethod
    def _build_linear_graph() -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2, label="b")
        graph.add_edge(2, 3, label="c")
        return graph

    def test_graph_to_nfa_with_explicit_start_and_final_states(self):
        graph = self._build_linear_graph()

        nfa = graph_to_nfa(graph, {0}, {3})

        assert isinstance(nfa, NondeterministicFiniteAutomaton)
        assert nfa.accepts(["a", "b", "c"])
        assert not nfa.accepts(["a", "b"])
        assert not nfa.accepts(["a", "b", "c", "d"])

    def test_graph_to_nfa_contains_all_graph_nodes_as_states(self):
        graph = self._build_linear_graph()

        nfa = graph_to_nfa(graph, {0}, {3})

        assert len(nfa.states) == graph.number_of_nodes()

    def test_graph_to_nfa_default_start_and_final_states(self):
        graph = self._build_linear_graph()

        nfa = graph_to_nfa(graph, set(), set())

        assert len(nfa.start_states) == graph.number_of_nodes()
        assert len(nfa.final_states) == graph.number_of_nodes()
        assert nfa.accepts(["b", "c"])
        assert nfa.accepts(["c"])
        assert nfa.accepts([])

    def test_graph_to_nfa_multiple_start_and_final_states(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="x")
        graph.add_edge(2, 1, label="y")

        nfa = graph_to_nfa(graph, {0, 2}, {1})

        assert nfa.accepts(["x"])
        assert nfa.accepts(["y"])
        assert not nfa.accepts(["z"])

    def test_graph_to_nfa_ignores_edges_without_label(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_edge(1, 2)  # no label attribute

        nfa = graph_to_nfa(graph, {0}, {2})

        assert not nfa.accepts(["a"])

    def test_graph_to_nfa_isolated_node_is_included_as_state(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(0, 1, label="a")
        graph.add_node(2)

        nfa = graph_to_nfa(graph, {0}, {1})

        assert len(nfa.states) == 3

    def test_graph_to_nfa_empty_graph(self):
        graph = nx.MultiDiGraph()

        nfa = graph_to_nfa(graph, set(), set())

        assert len(nfa.states) == 0
        assert len(nfa.start_states) == 0
        assert len(nfa.final_states) == 0
