from typing import NamedTuple, Any
import cfpq_data
import networkx as nx
from networkx.drawing.nx_pydot import write_dot


class GraphInfo(NamedTuple):
    number_of_nodes: int
    number_of_edges: int
    labels: set[str]


def get_graph_info(graph_name: str) -> GraphInfo:
    graph_path = cfpq_data.download(graph_name)
    graph = cfpq_data.graph_from_csv(graph_path)

    nodes_cnt = graph.number_of_nodes()
    edges_cnt = graph.number_of_edges()
    labels = set(cfpq_data.get_sorted_labels(graph))

    return GraphInfo(nodes_cnt, edges_cnt, labels)


def save_two_cycles_graph(
    n: int,
    m: int,
    labels: tuple[str, str],
    output_path: str,
    common_node: int | Any = 0,
) -> nx.MultiDiGraph:
    graph = cfpq_data.labeled_two_cycles_graph(
        n, m, common_node=common_node, labels=labels
    )

    pydot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    pydot_graph.write_raw(output_path)
    return graph
