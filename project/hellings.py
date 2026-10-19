from pyformlang.cfg import CFG, Variable, Terminal, Production, Epsilon


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

            if variable not in used_variables:
                used_variables.add(variable)
                return variable

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
