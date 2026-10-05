from networkx import MultiDiGraph
from scipy.sparse import csr_matrix, identity, kron, lil_matrix

from project.automata import graph_to_nfa, regex_to_dfa
from project.rpq import AdjacencyMatrixFA


def ms_bfs_based_rpq(
        regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    if not regex_fa.start_states or not graph_fa.start_states:
        return set()

    starts = sorted(graph_fa.start_states)
    regex_start = next(iter(regex_fa.start_states))
    n_regex = regex_fa.num_states
    n_graph = graph_fa.num_states

    front = lil_matrix((len(starts) * n_regex, n_graph), dtype=bool)
    for i, start in enumerate(starts):
        front[i * n_regex + regex_start, start] = True
    front = front.tocsr()
    visited = front

    steps: list[tuple[csr_matrix, csr_matrix]] = [
        (
            kron(
                identity(len(starts), dtype=bool),
                regex_fa.matrices[symbol].T,
                format="csr",
            ).astype(bool),
            graph_fa.matrices[symbol],
        )
        for symbol in regex_fa.matrices.keys() & graph_fa.matrices.keys()
    ]

    while front.nnz > 0:
        new_front = csr_matrix(front.shape, dtype=bool)
        for regex_step, graph_step in steps:
            new_front = new_front + regex_step @ (front @ graph_step)
        front = new_front > visited
        visited = visited + front

    result = set()
    for i, start in enumerate(starts):
        for regex_final in regex_fa.final_states:
            _, reached = visited[i * n_regex + regex_final].nonzero()
            for v in reached:
                if v in graph_fa.final_states:
                    result.add(
                        (
                            graph_fa.index_to_state[start].value,
                            graph_fa.index_to_state[v].value,
                        )
                    )
    return result
