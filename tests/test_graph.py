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
