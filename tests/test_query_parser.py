import pytest
from antlr4 import ParserRuleContext

from project.query_parser import program_to_tree, tree_to_program, nodes_count


VALID_PROGRAMS = [
    "",
    " \t\n",
    "let g is graph",
    "let x = 0",
    "let x = 123",
    'let x = "a"',
    "let x = y",
    "let x = [1, 2, 3]",
    'let x = (1, "a", 2)',
    "add vertex 1 to g",
    'add edge (1, "a", 2) to g',
    "remove vertex 1 from g",
    "remove vertices [1, 2] from g",
    'remove edge (1, "a", 2) from g',
    'let q = ("a" | "b") . "c" & "d"',
    'let q = "a"^[0..]',
    'let q = "a"^[1..3]',
    "let r = return u where u reachable from v in g by q",
    "let r = for v in [1] return u where u reachable from v in g by q",
    "let r = for v in [1] for u in [2, 3] return u,v "
    "where u reachable from v in g by q",
    'let g is graph\nadd edge (1, "a", 2) to g\n'
    'let q = "a" . q | "b"\n'
    "let r = for v in [1] return u where u reachable from v in g by q",
]


class TestProgramToTree:
    @pytest.mark.parametrize("program", VALID_PROGRAMS)
    def test_valid_program(self, program):
        tree, valid = program_to_tree(program)
        assert valid
        assert isinstance(tree, ParserRuleContext)

    @pytest.mark.parametrize(
        "program",
        [
            "let x 1",
            "let g is",
            "let x =",
            "let x = [1 2]",
            "let x = []",
            'add edge (1, "a") to g',
            "add vertex 1 from g",
            "remove vertex 1 to g",
            'let q = "a"^[..3]',
            'let q = "a" |',
            "let r = return u where u reachable from v in g",
            "let g is graph trailing",
        ],
    )
    def test_syntax_error(self, program):
        _, valid = program_to_tree(program)
        assert not valid

    @pytest.mark.parametrize(
        "program",
        ["let g is graph @", "@", 'let x = "ab"', "let x = -1"],
    )
    def test_lexer_error(self, program):
        _, valid = program_to_tree(program)
        assert not valid


class TestNodesCount:
    @pytest.mark.parametrize(
        "program, expected", [("", 2), ("let g is graph", 8), ("let x = 1", 9)]
    )
    def test_rules_and_terminals_including_eof(self, program, expected):
        tree, valid = program_to_tree(program)
        assert valid
        assert nodes_count(tree) == expected
        assert nodes_count(tree) == expected

    def test_whitespace_does_not_add_nodes(self):
        compact, _ = program_to_tree("let x=1")
        spaced, _ = program_to_tree("\nlet\t x = 1\n")
        assert nodes_count(compact) == nodes_count(spaced)


class TestTreeToProgram:
    def test_simple_binding(self):
        tree, valid = program_to_tree("let x = 1")
        assert valid
        assert tree_to_program(tree) == "let x = 1"

    @pytest.mark.parametrize("program", VALID_PROGRAMS)
    def test_round_trip(self, program):
        tree, valid = program_to_tree(program)
        assert valid
        restored = tree_to_program(tree)
        assert "<EOF>" not in restored
        restored_tree, restored_valid = program_to_tree(restored)
        assert restored_valid
        assert restored_tree.toStringTree(recog=None) == tree.toStringTree(recog=None)
        assert nodes_count(restored_tree) == nodes_count(tree)
        assert tree_to_program(restored_tree) == restored
