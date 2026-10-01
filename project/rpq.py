from itertools import product
from typing import Iterable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import csr_matrix, identity, kron, lil_matrix

from project.automata import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton | None = None):
        self.matrices: dict[Symbol, csr_matrix] = {}
        self.index_to_state: dict[int, State] = {}
        self.state_to_index: dict[State, int] = {}
        self.start_states: set[int] = set()
        self.final_states: set[int] = set()
        self.num_states: int = 0

        if automaton is None:
            return

        self.index_to_state = dict(enumerate(automaton.states))
        self.state_to_index = {st: i for i, st in self.index_to_state.items()}
        self.num_states = len(self.index_to_state)
        self.start_states = {self.state_to_index[st] for st in automaton.start_states}
        self.final_states = {self.state_to_index[st] for st in automaton.final_states}

        builders: dict[Symbol, lil_matrix] = {}
        for from_state, transitions in automaton.to_dict().items():
            for symbol, to_states in transitions.items():
                if not isinstance(to_states, set):
                    to_states = {to_states}
                if symbol not in builders:
                    builders[symbol] = lil_matrix(
                        (self.num_states, self.num_states), dtype=bool
                    )
                for to_state in to_states:
                    builders[symbol][
                        self.state_to_index[from_state], self.state_to_index[to_state]
                    ] = True

        self.matrices = {symbol: m.tocsr() for symbol, m in builders.items()}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current = lil_matrix((1, self.num_states), dtype=bool)
        for i in self.start_states:
            current[0, i] = True
        current = current.tocsr()

        for symbol in word:
            if not isinstance(symbol, Symbol):
                symbol = Symbol(symbol)
            if symbol not in self.matrices:
                return False
            current = current @ self.matrices[symbol]
            if current.nnz == 0:
                return False

        _, reached = current.nonzero()
        return any(i in self.final_states for i in reached)

    def transitive_closure(self) -> csr_matrix:
        closure = identity(self.num_states, dtype=bool, format="csr")
        for m in self.matrices.values():
            closure = closure + m

        while True:
            prev_nnz = closure.nnz
            closure = closure + closure @ closure
            if closure.nnz == prev_nnz:
                return closure

    def is_empty(self) -> bool:
        closure = self.transitive_closure()
        return not any(
            closure[start, final]
            for start, final in product(self.start_states, self.final_states)
        )


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    n2 = automaton2.num_states
    result = AdjacencyMatrixFA()
    result.num_states = automaton1.num_states * n2

    for symbol in automaton1.matrices.keys() & automaton2.matrices.keys():
        result.matrices[symbol] = kron(
            automaton1.matrices[symbol], automaton2.matrices[symbol], format="csr"
        ).astype(bool)

    for i, j in product(
        automaton1.index_to_state.keys(), automaton2.index_to_state.keys()
    ):
        state = State((automaton1.index_to_state[i], automaton2.index_to_state[j]))
        result.index_to_state[i * n2 + j] = state
        result.state_to_index[state] = i * n2 + j

    result.start_states = {
        i * n2 + j for i, j in product(automaton1.start_states, automaton2.start_states)
    }
    result.final_states = {
        i * n2 + j for i, j in product(automaton1.final_states, automaton2.final_states)
    }
    return result


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    intersection = intersect_automata(graph_fa, regex_fa)
    closure = intersection.transitive_closure()

    n2 = regex_fa.num_states
    result = set()
    for start, final in product(intersection.start_states, intersection.final_states):
        if closure[start, final]:
            u = graph_fa.index_to_state[start // n2].value
            v = graph_fa.index_to_state[final // n2].value
            result.add((u, v))
    return result
