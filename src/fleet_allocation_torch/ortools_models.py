"""The two thesis optimisation models, ported from ``docplex`` to OR-Tools CP-SAT.

Allocation (Section 4.1.3 of the thesis). One boolean per SEC and AEV count, exactly one count per SEC, at
most ``num_EVs`` AEVs in total, maximise the summed look-up values, with ``ub_TPs`` as the upper bound of the
objective variable. The original required the total to equal ``num_EVs``. The look-up tables never decrease
with the AEV count, so the two optima coincide and the port uses "at most", which stays feasible when the
per-SEC range is capped.

Colouring (the negotiation schedule). One integer colour per node, adjacent nodes differ, minimise the number
of colours, with the original symmetry breaking: the first node that has a neighbour takes colour 0 and its
first neighbour colour 1, and isolated nodes take colour 0.

The two functions with the original names keep the original signatures and return values. The ``solve_*``
helpers underneath take arrays and return a result record with the bound, gap, wall time and status.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from ortools.sat.python import cp_model

STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


@dataclass
class SolveResult:
    objective: float
    bound: float
    gap: float
    wall_s: float
    status: str
    solution: list[int]

    def as_dict(self) -> dict:
        return asdict(self)


def _gap(objective: float, bound: float) -> float:
    if objective == 0:
        return 0.0 if bound == 0 else 1.0
    return abs(bound - objective) / abs(objective)


def solve_allocation(values: np.ndarray, num_evs: int, ub: int | None = None, time_limit: float = 60.0, workers: int = 8,
                     exact_total: bool = False) -> SolveResult:
    """Maximise the summed look-up values. ``values`` has one row per SEC and one column per AEV count from 0."""
    values = np.asarray(values, dtype=np.int64)
    num_secs, width = values.shape
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{c}_{j}") for j in range(width)] for c in range(num_secs)]
    for c in range(num_secs):
        model.AddExactlyOne(x[c])
    total = sum(j * x[c][j] for c in range(num_secs) for j in range(width))
    if exact_total:
        model.Add(total == num_evs)
    else:
        model.Add(total <= num_evs)
    ub = int(values.max(axis=1).sum()) if ub is None else int(ub)
    obj = model.NewIntVar(0, ub, "obj")
    model.Add(obj == sum(int(values[c, j]) * x[c][j] for c in range(num_secs) for j in range(width)))
    model.Maximize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    wall = time.perf_counter() - t0
    name = STATUS_NAMES.get(status, str(status))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        alloc = [next(j for j in range(width) if solver.Value(x[c][j])) for c in range(num_secs)]
        objective = float(solver.ObjectiveValue())
        bound = float(solver.BestObjectiveBound())
    else:
        alloc, objective, bound = [], 0.0, float(ub)
    return SolveResult(objective, bound, _gap(objective, bound), wall, name, alloc)


def symmetric_neighbours(num_nodes: int, adjacency: list[list[int]]) -> list[list[int]]:
    """Neighbour lists with every edge present on both sides, sorted, without self loops. The adjacency lists
    a caller passes may be one sided (an edge listed under one endpoint only), and the model must not read a
    node with an empty list as isolated when another node lists it."""
    out: list[set[int]] = [set() for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in adjacency[i]:
            if i != j:
                out[i].add(j)
                out[j].add(i)
    return [sorted(s) for s in out]


def greedy_colouring(num_nodes: int, neighbours: list[list[int]]) -> list[int]:
    """First-fit colouring in node order. Its colour count is a valid upper bound for the minimum."""
    colours = [-1] * num_nodes
    for i in range(num_nodes):
        taken = {colours[j] for j in neighbours[i] if colours[j] >= 0}
        c = 0
        while c in taken:
            c += 1
        colours[i] = c
    return colours


def solve_colouring(num_nodes: int, adjacency: list[list[int]], max_colours: int | None = None, time_limit: float = 60.0,
                    workers: int = 8) -> SolveResult:
    """Minimise the number of colours on a graph given as adjacency lists.

    ``max_colours`` bounds the colour domain. When it is not given, the first-fit colouring's count is used,
    which is an upper bound on the minimum and keeps the domain small on large graphs. The original model took
    the bound from a previous solution in the same way."""
    neighbours = symmetric_neighbours(num_nodes, adjacency)
    if max_colours is None:
        max_colours = max(greedy_colouring(num_nodes, neighbours), default=-1) + 1
    max_colours = max(int(max_colours), 1 if num_nodes else 0)
    model = cp_model.CpModel()
    colour = [model.NewIntVar(0, max(max_colours - 1, 0), f"c_{i}") for i in range(num_nodes)]
    edges = {(i, j) for i in range(num_nodes) for j in neighbours[i] if i < j}
    for i, j in sorted(edges):
        model.Add(colour[i] != colour[j])
    first = next((i for i in range(num_nodes) if neighbours[i]), None)
    if first is not None:
        model.Add(colour[first] == 0)
        model.Add(colour[neighbours[first][0]] == 1)
    for i in range(num_nodes):
        if not neighbours[i]:
            model.Add(colour[i] == 0)
    used = model.NewIntVar(1 if num_nodes else 0, max(max_colours, 1), "colours")
    for i in range(num_nodes):
        model.Add(used >= colour[i] + 1)
    model.Minimize(used)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    wall = time.perf_counter() - t0
    name = STATUS_NAMES.get(status, str(status))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [int(solver.Value(c)) for c in colour]
        objective = float(solver.ObjectiveValue())
        bound = float(solver.BestObjectiveBound())
    else:
        sol, objective, bound = [], 0.0, 0.0
    return SolveResult(objective, bound, _gap(objective, bound), wall, name, sol)


def solve_compute_optimal_EV_2_SEC_allocation(ub_TPs, num_SECs, num_EVs, array_values, time_limit):
    """Original signature. ``array_values`` concatenates, SEC by SEC, the look-up values for 0 to ``num_EVs``
    AEVs. Returns ``(succeed, best_value, EV_per_SEC)`` with ``succeed`` 1 when optimal, 0 when a solution was
    found without proof, and -1 when none was found."""
    values = np.asarray(array_values, dtype=np.int64).reshape(num_SECs, num_EVs + 1)
    res = solve_allocation(values, num_EVs, ub=ub_TPs, time_limit=time_limit, exact_total=True)
    if res.status == "OPTIMAL":
        return 1, int(res.objective), list(res.solution)
    if res.status == "FEASIBLE":
        return 0, int(res.objective), list(res.solution)
    return -1, -1, [0] * num_SECs


def solve_compute_optimal_SEC_negotiation_schedule(num_nodes, edges_per_node, best_value, time_limit):
    """Original signature. ``edges_per_node[node]`` is ``(count, [neighbour nodes])`` and ``best_value`` is a
    colour count to improve on. Returns ``(succeed, best_value, colour_per_node)``."""
    adjacency = [list(edges_per_node[i][1]) for i in range(num_nodes)]
    res = solve_colouring(num_nodes, adjacency, max_colours=max(best_value - 1, 1), time_limit=time_limit)
    if res.status == "OPTIMAL":
        return 1, int(res.objective), list(res.solution)
    if res.status == "FEASIBLE":
        return 0, int(res.objective), list(res.solution)
    return -1, best_value, [-1] * num_nodes


def line_graph(num_secs: int, connections: list[tuple[int, int]]) -> tuple[int, list[list[int]]]:
    """The negotiation graph: one node per SEC connection, two nodes adjacent when the connections share a
    SEC. Colouring it schedules the pairwise negotiations into rounds, as the thesis step 3 does."""
    by_sec: dict[int, list[int]] = {}
    for idx, (a, b) in enumerate(connections):
        by_sec.setdefault(a, []).append(idx)
        by_sec.setdefault(b, []).append(idx)
    adjacency: list[set[int]] = [set() for _ in connections]
    for members in by_sec.values():
        for i in members:
            for j in members:
                if i != j:
                    adjacency[i].add(j)
    return len(connections), [sorted(s) for s in adjacency]
