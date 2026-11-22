from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker, ParserRuleContext
from project.generated.GraphQueryLexer import GraphQueryLexer
from project.generated.GraphQueryListener import GraphQueryListener
from project.generated.GraphQueryParser import GraphQueryParser
from antlr4.error.ErrorListener import ErrorListener


class SyntaxErrorListener(ErrorListener):
    def __init__(self):
        super().__init__()
        self.has_errors = False

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.has_errors = True


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    errors = SyntaxErrorListener()

    lexer = GraphQueryLexer(InputStream(program))
    lexer.removeErrorListeners()
    lexer.addErrorListener(errors)

    parser = GraphQueryParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser.addErrorListener(errors)

    tree = parser.prog()
    return tree, not errors.has_errors


class NodeCountListener(GraphQueryListener):
    def __init__(self):
        self.count = 0

    def enterEveryRule(self, ctx):
        self.count += 1

    def visitTerminal(self, node):
        self.count += 1

    def visitErrorNode(self, node):
        self.count += 1


def nodes_count(tree: ParserRuleContext) -> int:
    listener = NodeCountListener()
    ParseTreeWalker().walk(listener, tree)
    return listener.count


def tree_to_program(tree: ParserRuleContext) -> str:
    return tree.getText()
