from collections import defaultdict, deque

import networkx as nx
from pyformlang.cfg import CFG, Variable, Terminal, Production


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    # isinstance(object, Class)
    used_variables = set(cfg.variables)
    terminal_variables = {}
    counter = 0

    def fresh_variable():
        nonlocal counter

        while True:
            var = Variable(f"._.{counter}")
            counter += 1

            if var not in used_variables:
                used_variables.add(var)
                return var

    new_cfg = cfg.eliminate_unit_productions()
    new_rules = set()

    for production in new_cfg.productions:
        head = production.head
        body = list(production.body)

        if len(body) <= 1:
            new_rules.add(production)
            continue

        for index, symbol in enumerate(body):
            if isinstance(symbol, Terminal):
                if symbol not in terminal_variables:
                    variable = fresh_variable()
                    terminal_variables[symbol] = variable
                    new_rules.add(Production(variable, [symbol]))

                body[index] = terminal_variables[symbol]

        while len(body) > 2:
            variable = fresh_variable()
            new_rules.add(Production(head, [body[0], variable]))
            head = variable
            body = body[1:]

        new_rules.add(Production(head, body))

    return CFG(start_symbol=new_cfg.start_symbol, productions=new_rules)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    my_cfg = cfg_to_weak_normal_form(cfg)
    terminal_rules = defaultdict(set)
    left_rules = defaultdict(set)
    right_rules = defaultdict(set)
    nullable = set()

    for production in my_cfg.productions:
        body = production.body
        if not body:
            nullable.add(production.head)
        elif len(body) == 1:
            terminal_rules[body[0]].add(production.head)
        else:
            first, second = body
            left_rules[first].add((production.head, second))
            right_rules[second].add((production.head, first))

    reach = set()
    queue = deque()
    outgoing = defaultdict(set)
    incoming = defaultdict(set)

    def add_path(var, start, end):
        triple = (var, start, end)
        if triple not in reach:
            reach.add(triple)
            outgoing[var, start].add(end)
            incoming[var, end].add(start)
            queue.append(triple)

    for start, end, data in graph.edges(data=True):
        terminal = Terminal(data["label"])
        for var in terminal_rules[terminal]:
            add_path(var, start, end)

    for var in nullable:
        for node in graph.nodes:
            add_path(var, node, node)

    while queue:
        var, start, end = queue.popleft()
        for head, second in left_rules[var]:
            for final in tuple(outgoing[second, end]):
                add_path(head, start, final)

        for head, first in right_rules[var]:
            for initial in tuple(incoming[first, start]):
                add_path(head, initial, end)

    return set(
        (u, v)
        for var, u, v in reach
        if (
            var == my_cfg.start_symbol
            and (start_nodes is None or u in start_nodes)
            and (final_nodes is None or v in final_nodes)
        )
    )
