from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    e_nfa = Regex(regex).to_epsilon_nfa()
    dfa = e_nfa.to_deterministic()
    return dfa.minimize()

