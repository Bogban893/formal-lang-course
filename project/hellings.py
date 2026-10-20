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
    reach = set()
    for start, end, data in graph.edges(data=True):
        terminal = Terminal(data["label"])
        for production in my_cfg.productions:
            if len(production.body) == 1 and production.body[0] == terminal:
                triple = (production.head, start, end)
                reach.add(triple)

    for production in my_cfg.productions:
        if len(production.body) == 0:
            for node in graph.nodes:
                reach.add((production.head, node, node))

    while True:
        new_reach = set()

        for production in my_cfg.productions:
            if len(production.body) != 2:
                continue

            head = production.head
            first, second = production.body

            for var1, u, k in reach:
                for var2, k2, v in reach:
                    if var1 == first and var2 == second and k == k2:
                        new_reach.add((head, u, v))

        if new_reach.issubset(reach):
            break

        reach.update(new_reach)

    return set(
        (u, v)
        for var, u, v in reach
        if (
            var == my_cfg.start_symbol
            and (start_nodes is None or u in start_nodes)
            and (final_nodes is None or v in final_nodes)
        )
    )
