from collections import defaultdict, deque

import networkx as nx
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    nodes = set(graph.nodes)
    starts = nodes if start_nodes is None else nodes & set(start_nodes)
    finals = nodes if final_nodes is None else nodes & set(final_nodes)
    boxes, initial = rsm.boxes, rsm.initial_label
    if not starts or not finals or initial not in boxes:
        return set()

    edges = defaultdict(set)
    for u, v, label in graph.edges(data="label"):
        edges[u, Symbol(label)].add(v)

    moves = {
        (label, state): [
            (symbol, t)
            for symbol, targets in out.items()
            for t in (targets if isinstance(targets, set) else {targets})
        ]
        for label, box in boxes.items()
        for state, out in box.dfa.to_dict().items()
    }

    queue, seen = deque(), set()
    callers = defaultdict(set)
    completed = defaultdict(set)
    result = set()

    def schedule(state, stack, vertex):
        descriptor = (state, stack, vertex)
        if descriptor not in seen:
            seen.add(descriptor)
            queue.append(descriptor)

    def enter(stack):
        label, vertex = stack
        for state in boxes[label].dfa.start_states:
            schedule(state, stack, vertex)

    for vertex in starts:
        enter((initial, vertex))

    while queue:
        state, stack, vertex = queue.popleft()
        label, entry = stack

        if state in boxes[label].dfa.final_states:
            if label == initial and entry in starts and vertex in finals:
                result.add((entry, vertex))
            if vertex not in completed[stack]:
                completed[stack].add(vertex)
                for return_state, parent in callers[stack]:
                    schedule(return_state, parent, vertex)

        for symbol, target in moves.get((label, state), ()):
            if symbol in boxes:
                callee = (symbol, vertex)
                if (target, stack) not in callers[callee]:
                    callers[callee].add((target, stack))
                    for end in completed[callee]:
                        schedule(target, stack, end)
                enter(callee)
            else:
                for end in edges[vertex, symbol]:
                    schedule(target, stack, end)

    return result
