import networkx as nx
import pytest
from pyformlang.cfg import CFG

from project.tensor_cfpq import cfg_to_rsm, ebnf_to_rsm, tensor_based_cfpq


class TestTensorBasedCFPQ:
    @pytest.mark.parametrize(
        "convert",
        [
            lambda: cfg_to_rsm(CFG.from_text("S -> a S b | epsilon")),
            lambda: ebnf_to_rsm("S -> a S b | epsilon"),
        ],
    )
    def test_recursive_query_and_filters(self, convert):
        graph = nx.MultiDiGraph()
        graph.add_edge(10, 20, label="x")
        graph.add_edge(10, 20, label="a")
        graph.add_edge(20, 30, label="b")
        graph.add_node(99)
        rsm = convert()

        assert tensor_based_cfpq(rsm, graph) == {
            (10, 10),
            (20, 20),
            (30, 30),
            (99, 99),
            (10, 30),
        }
        assert tensor_based_cfpq(rsm, graph, {10}, {30}) == {(10, 30)}
        assert tensor_based_cfpq(rsm, graph, set(), None) == set()
        assert tensor_based_cfpq(rsm, graph, None, set()) == set()
        assert tensor_based_cfpq(rsm, nx.DiGraph()) == set()

    def test_empty_language(self):
        graph = nx.DiGraph()
        graph.add_node(42)

        assert tensor_based_cfpq(cfg_to_rsm(CFG()), graph) == set()
