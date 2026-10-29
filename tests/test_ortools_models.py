"""CP-SAT ports against brute force: allocation with 3 SECs and 6 AEVs, colouring on 5-node graphs."""
import itertools

import numpy as np

from fleet_allocation_torch.ortools_models import (
    greedy_colouring,
    line_graph,
    solve_allocation,
    solve_colouring,
    solve_compute_optimal_EV_2_SEC_allocation,
    solve_compute_optimal_SEC_negotiation_schedule,
    symmetric_neighbours,
)


def brute_force_allocation(values, num_evs, exact):
    num_secs, width = values.shape
    best = -1
    for combo in itertools.product(range(width), repeat=num_secs):
        total = sum(combo)
        if (exact and total != num_evs) or (not exact and total > num_evs):
            continue
        best = max(best, sum(int(values[c, j]) for c, j in enumerate(combo)))
    return best


def monotone_tables(seed, num_secs=3, num_evs=6):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_secs):
        gains = rng.integers(0, 10, size=num_evs)
        rows.append(np.concatenate([[0], np.cumsum(gains)]))
    return np.array(rows, dtype=np.int64)


def test_allocation_matches_brute_force_on_random_tables():
    mismatches = 0
    for seed in range(12):
        values = monotone_tables(seed)
        res = solve_allocation(values, 6, time_limit=10, workers=4)
        assert res.status == "OPTIMAL"
        if int(res.objective) != brute_force_allocation(values, 6, exact=False):
            mismatches += 1
        assert sum(res.solution) <= 6
        assert sum(int(values[c, j]) for c, j in enumerate(res.solution)) == int(res.objective)
    assert mismatches == 0


def test_original_signature_allocation_matches_brute_force_with_exact_total():
    mismatches = 0
    for seed in range(8):
        values = monotone_tables(seed)
        array_values = values.reshape(-1).tolist()
        succeed, best_value, ev_per_sec = solve_compute_optimal_EV_2_SEC_allocation(int(values.max(axis=1).sum()), 3, 6, array_values, 10)
        assert succeed == 1
        assert sum(ev_per_sec) == 6
        if best_value != brute_force_allocation(values, 6, exact=True):
            mismatches += 1
    assert mismatches == 0


def edges_of(num_nodes, adjacency):
    return {(min(i, j), max(i, j)) for i in range(num_nodes) for j in adjacency[i] if i != j}


def chromatic_number(num_nodes, adjacency):
    edges = edges_of(num_nodes, adjacency)
    for k in range(1, num_nodes + 1):
        for colours in itertools.product(range(k), repeat=num_nodes):
            if all(colours[i] != colours[j] for i, j in edges):
                return k
    return num_nodes


GRAPHS = {
    "cycle5": [[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]],
    "path5": [[1], [0, 2], [1, 3], [2, 4], [3]],
    "clique5": [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]],
    "star_plus_edge": [[1, 2, 3, 4], [0, 2], [0, 1], [0], [0]],
    "with_isolated": [[1], [0, 2], [1, 4], [], [2]],
    "one_sided_lists": [[1, 2], [], [3], [], [0]],
}


def test_colouring_matches_brute_force_on_five_node_graphs():
    mismatches = 0
    for name, adj in GRAPHS.items():
        res = solve_colouring(5, adj, time_limit=10, workers=4)
        assert res.status == "OPTIMAL", name
        if int(res.objective) != chromatic_number(5, adj):
            mismatches += 1
        for i, j in edges_of(5, adj):
            assert res.solution[i] != res.solution[j], name
    assert mismatches == 0


def test_original_signature_colouring():
    adj = GRAPHS["cycle5"]
    edges_per_node = {i: (len(adj[i]), list(adj[i])) for i in range(5)}
    succeed, best_value, colours = solve_compute_optimal_SEC_negotiation_schedule(5, edges_per_node, 5, 10)
    assert succeed == 1
    assert best_value == 3
    assert colours[0] == 0 and colours[1] == 1


def test_line_graph_of_a_path_of_secs():
    n, adj = line_graph(4, [(0, 1), (1, 2), (2, 3)])
    assert n == 3
    assert adj == [[1], [0, 2], [1]]


def test_symmetric_neighbours_and_greedy_bound_on_one_sided_lists():
    adj = GRAPHS["one_sided_lists"]
    nb = symmetric_neighbours(5, adj)
    assert nb == [[1, 2, 4], [0], [0, 3], [2], [0]]
    greedy = greedy_colouring(5, nb)
    for i, j in edges_of(5, adj):
        assert greedy[i] != greedy[j]
    assert max(greedy) + 1 >= chromatic_number(5, adj)
