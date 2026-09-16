import networkx as nx
import pydot
import pytest

from project.graph import get_graph_info, save_two_cycles_graph, GraphInfo


class TestGraphMetadata:
    def test_get_graph_info_valid_dataset(self):
        info = get_graph_info("skos")

        assert isinstance(info, GraphInfo)
        assert info.number_of_nodes == 144
        assert info.number_of_edges == 252
        assert "type" in info.labels
        assert "label" in info.labels
        assert "definition" in info.labels

    @pytest.mark.parametrize("invalid_name", ["non_existent_graph_123", ""])
    def test_get_graph_info_raises_on_invalid_name(self, invalid_name):
        with pytest.raises(Exception):
            get_graph_info(invalid_name)


class TestTwoCyclesGraphGeneration:
    def test_save_two_cycles_graph_structure_and_dot_export(self, tmp_path):
        output_file = tmp_path / "custom_cycles.dot"
        n, m = 4, 3
        labels = ("alpha", "beta")
        center_node = 100

        graph = save_two_cycles_graph(
            n, m, labels, str(output_file), common_node=center_node
        )
        expected_nodes_count = n + m + 1
        expected_edges_count = (n + 1) + (m + 1)

        assert graph.number_of_nodes() == expected_nodes_count
        assert graph.number_of_edges() == expected_edges_count
        assert center_node in graph.nodes

        edge_labels = {data for _, _, data in graph.edges(data="label")}
        assert edge_labels == set(labels)

        parsed_graphs = pydot.graph_from_dot_file(str(output_file))
        assert parsed_graphs is not None
        assert len(parsed_graphs) == 1

        dot_graph = parsed_graphs[0]
        assert dot_graph.get_type() == "digraph"

        dot_nodes = {node.get_name() for node in dot_graph.get_nodes()}
        assert str(center_node) in dot_nodes
        assert len(dot_graph.get_edges()) == expected_edges_count

    def test_save_two_cycles_graph_custom_defaults(self, tmp_path):
        target_path = tmp_path / "default_node.dot"
        n, m = 1, 1

        graph = save_two_cycles_graph(n, m, ("l1", "l2"), str(target_path))

        assert 0 in graph.nodes
        assert graph.number_of_nodes() == 3
        assert target_path.exists()

    def test_save_two_cycles_graph_invalid_input_file_integrity(self, tmp_path):
        file_path = tmp_path / "protected.dot"
        initial_data = b"digraph { dummy -> node; }"
        file_path.write_bytes(initial_data)

        with pytest.raises((ValueError, nx.NetworkXError)):
            save_two_cycles_graph(-2, 3, ("a", "b"), str(file_path))

        assert file_path.read_bytes() == initial_data
