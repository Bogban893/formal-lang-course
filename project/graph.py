from typing import NamedTuple, Any
import cfpq_data



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



