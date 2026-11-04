from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Variable
from pyformlang.finite_automaton import EpsilonNFA, State, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_matrix, identity, kron, lil_matrix


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    start_symbol = cfg.start_symbol if cfg.start_symbol is not None else Variable("S")
    automata = {variable: EpsilonNFA() for variable in cfg.variables | {start_symbol}}
    for automaton in automata.values():
        automaton.add_start_state(State("start"))

    for index, prod in enumerate(cfg.productions):
        automaton = automata[prod.head]
        current = State("start")
        for position, symbol in enumerate(prod.body):
            target = State((index, position))
            automaton.add_transition(current, Symbol(symbol.value), target)
            current = target
        automaton.add_final_state(current)

    boxes = {Box(automaton, Symbol(var.value)) for var, automaton in automata.items()}
    return RecursiveAutomaton(initial_label=Symbol(start_symbol.value), boxes=boxes)


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    nodes = list(graph.nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    if size == 0:
        return set()

    graph_matrices = defaultdict(lambda: lil_matrix((size, size), dtype=bool))
    for start, end, data in graph.edges(data=True):
        graph_matrices[Symbol(data["label"])][
            node_to_index[start], node_to_index[end]
        ] = True
    for label in rsm.labels:
        graph_matrices[label]
    graph_matrices = {label: matrix.tocsr() for label, matrix in graph_matrices.items()}

    state_to_index = {}
    for label, box in rsm.boxes.items():
        for state in box.dfa.states:
            state_to_index[label, state] = len(state_to_index)
    rsm_size = len(state_to_index)

    rsm_matrices = defaultdict(lambda: lil_matrix((rsm_size, rsm_size), dtype=bool))
    boundaries = []
    for label, box in rsm.boxes.items():
        for source, transitions in box.dfa.to_dict().items():
            for symbol, targets in transitions.items():
                if not isinstance(targets, set):
                    targets = {targets}
                for target in targets:
                    rsm_matrices[symbol][
                        state_to_index[label, source], state_to_index[label, target]
                    ] = True
        for start in box.dfa.start_states:
            for final in box.dfa.final_states:
                boundaries.append(
                    (label, state_to_index[label, start], state_to_index[label, final])
                )
    rsm_matrices = {label: matrix.tocsr() for label, matrix in rsm_matrices.items()}

    changed = True
    while changed:
        changed = False

        product = identity(rsm_size * size, dtype=bool, format="csr")
        for label in rsm_matrices.keys() & graph_matrices.keys():
            product = product + kron(
                rsm_matrices[label], graph_matrices[label], format="csr"
            )

        closure = product
        while True:
            updated = closure + closure @ closure
            if updated.nnz == closure.nnz:
                break
            closure = updated

        for label, start, final in boundaries:
            paths = closure[
                start * size : (start + 1) * size,
                final * size : (final + 1) * size,
            ]
            updated = graph_matrices[label] + paths
            if updated.nnz != graph_matrices[label].nnz:
                graph_matrices[label] = updated
                changed = True

    result = graph_matrices.get(rsm.initial_label, csr_matrix((size, size), dtype=bool))
    rows, columns = result.nonzero()
    return {
        (nodes[row], nodes[column])
        for row, column in zip(rows, columns)
        if (start_nodes is None or nodes[row] in start_nodes)
        and (final_nodes is None or nodes[column] in final_nodes)
    }
