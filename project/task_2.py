from typing import Set

from networkx import MultiDiGraph
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    e_nfa = Regex(regex).to_epsilon_nfa()
    dfa = e_nfa.to_deterministic()
    return dfa.minimize()


def graph_to_nfa(
        graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for node in graph.nodes:
        nfa.states.add(State(node))

    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            nfa.add_transition(State(u), Symbol(label), State(v))

    actual_start = start_states if start_states else graph.nodes()
    for node in actual_start:
        nfa.add_start_state(State(node))

    actual_final = final_states if final_states else graph.nodes()
    for node in actual_final:
        nfa.add_final_state(State(node))

    return nfa
