import networkx as nx
from scipy.sparse import lil_matrix
from project.hellings import cfg_to_weak_normal_form
from pyformlang.cfg import CFG, Terminal


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    my_cfg = cfg_to_weak_normal_form(cfg)
    nodes = list(graph.nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)

    matrices = {
        variable: lil_matrix((size, size), dtype=bool)
        for variable in my_cfg.variables | {my_cfg.start_symbol}
    }
    binary_rules = []

    for prod in my_cfg.productions:
        head = prod.head
        body = prod.body

        # A -> a
        if len(body) == 1 and isinstance(body[0], Terminal):
            for start, end, data in graph.edges(data=True):
                if Terminal(data["label"]) == body[0]:
                    matrices[head][node_to_index[start], node_to_index[end]] = True

        # A -> epsilon
        elif len(body) == 0:
            matrices[head].setdiag(True)

        # A -> B C
        elif len(body) == 2:
            binary_rules.append((head, body[0], body[1]))

    matrices = {variable: matrix.tocsr() for variable, matrix in matrices.items()}

    changed = True
    while changed:
        changed = False
        for head, first, second in binary_rules:
            updated = matrices[head] + matrices[first] @ matrices[second]
            if updated.nnz != matrices[head].nnz:
                matrices[head] = updated
                changed = True

    rows, columns = matrices[my_cfg.start_symbol].nonzero()
    return {
        (nodes[row], nodes[column])
        for row, column in zip(rows, columns)
        if (start_nodes is None or nodes[row] in start_nodes)
        and (final_nodes is None or nodes[column] in final_nodes)
    }
