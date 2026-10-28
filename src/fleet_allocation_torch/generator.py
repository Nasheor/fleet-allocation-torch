"""Instance generator.

The thesis instance generator placed SECs on a square grid, linked each SEC to its eight grid neighbours in
turn until the requested number of connections was reached, and drew the trip petitions of each SEC from the
Google HashCode or NYC data. This module keeps the grid and the connection rule and replaces the trip data
with per-SEC look-up curves. A curve is a saturating function of the AEV count,

    values[j] = round(satisfiable * (1 - exp(-j / scale)))

whose ``satisfiable`` and ``scale`` are resampled, SEC by SEC, from a reference instance. The reference
curves come from the thesis simulator, so a generated instance has the same per-SEC distributions as the
thesis instance it is scaled from and ten times the SECs, AEVs and petitions.
"""
from __future__ import annotations

import math

import numpy as np

from .instances import Instance, cap_for


def grid_positions(num_secs: int, city: int) -> list[tuple[int, int]]:
    """SEC positions at the centres of a square grid of ``num_secs`` cells over a ``city`` by ``city`` map."""
    side = math.isqrt(num_secs)
    if side * side != num_secs:
        side = math.ceil(math.sqrt(num_secs))
    cell = city // side
    out = []
    for i in range(num_secs):
        r, c = divmod(i, side)
        out.append((c * cell + cell // 2, r * cell + cell // 2))
    return out


def grid_connections(num_secs: int, connections: int) -> list[tuple[int, int]]:
    """Connections added round robin over the SECs, each SEC taking its next unused grid neighbour.

    This is the rule of the thesis ``compute_neighbors.divide_and_compute_neighbors``. The candidate list of a
    SEC is its eight surrounding grid cells in the order north, south, west, east and the four diagonals.
    """
    side = math.ceil(math.sqrt(num_secs))
    candidates: list[list[int]] = []
    for i in range(num_secs):
        r, c = divmod(i, side)
        cand = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < side and 0 <= cc < side and rr * side + cc < num_secs:
                cand.append(rr * side + cc)
        candidates.append(cand)
    chosen: set[tuple[int, int]] = set()
    sec = 0
    stalls = 0
    while len(chosen) < connections and stalls < num_secs:
        added = False
        for n in candidates[sec]:
            edge = (min(sec, n), max(sec, n))
            if edge not in chosen:
                chosen.add(edge)
                added = True
                break
        stalls = 0 if added else stalls + 1
        sec = (sec + 1) % num_secs
    return sorted(chosen)


def fit_curve(row: np.ndarray) -> tuple[int, float]:
    """Fit ``satisfiable`` and ``scale`` of a saturating curve to one look-up row (index is the AEV count)."""
    satisfiable = int(row.max())
    if satisfiable <= 0:
        return 0, 1.0
    js = np.arange(1, len(row))
    frac = np.clip(row[1:] / satisfiable, 0.0, 0.999)
    mask = (row[1:] > 0) & (row[1:] < satisfiable)
    if mask.sum() < 2:
        return satisfiable, 1.0
    # least squares on -log(1 - frac) = j / scale, through the origin
    y = -np.log(1.0 - frac[mask])
    x = js[mask].astype(float)
    scale = float((x * x).sum() / max((x * y).sum(), 1e-9))
    return satisfiable, max(scale, 0.5)


def curve(satisfiable: int, scale: float, cap: int) -> np.ndarray:
    js = np.arange(cap + 1)
    return np.round(satisfiable * (1.0 - np.exp(-js / scale))).astype(np.int64)


def generate(name: str, reference: Instance, factor: int, seed: int, city: int = 10000) -> Instance:
    """Scale ``reference`` by ``factor`` with resampled per-SEC curves and the grid connection rule."""
    rng = np.random.default_rng(seed)
    num_secs = reference.num_secs * factor
    num_evs = reference.num_evs * factor
    num_petitions = reference.num_petitions * factor
    cap = cap_for(num_secs, num_evs)
    fits = [fit_curve(reference.values[s]) for s in range(reference.num_secs)]
    ref_petitions = reference.num_petitions / reference.num_secs
    ref_energy_per_ev = float(np.median([reference.energy[s, 1] for s in range(reference.num_secs) if reference.energy[s, 1] > 0] or [100.0]))
    picks = rng.integers(0, reference.num_secs, size=num_secs)
    sat = np.array([fits[p][0] for p in picks], dtype=float)
    scales = np.array([fits[p][1] for p in picks], dtype=float)
    # keep the share of satisfiable petitions of the reference instance at the new petition count
    target_total = reference.upper_bound() * factor
    sat = sat * (target_total / max(sat.sum(), 1.0))
    values = np.zeros((num_secs, cap + 1), dtype=np.int64)
    energy = np.zeros((num_secs, cap + 1), dtype=np.float64)
    for s in range(num_secs):
        values[s] = curve(int(round(sat[s])), scales[s], cap)
        energy[s] = np.arange(cap + 1) * ref_energy_per_ev * (1.0 + 0.1 * rng.standard_normal())
    energy = np.abs(energy)
    connections = grid_connections(num_secs, num_secs)
    return Instance(
        name=name,
        source=f"generated from {reference.name} with factor {factor}, seed {seed}",
        num_evs=num_evs,
        num_petitions=num_petitions,
        positions=grid_positions(num_secs, city),
        connections=connections,
        values=values,
        energy=energy,
        notes=(f"Per-SEC curves resampled from the {reference.num_secs} reference SECs and rescaled so that the "
               f"satisfiable share of petitions matches the reference ({reference.upper_bound()} of "
               f"{reference.num_petitions}). Mean petitions per SEC in the reference: {ref_petitions:.0f}."),
    )


def synthetic(name: str, num_secs: int, num_evs: int, num_petitions: int, seed: int, connections: int | None = None,
              city: int = 1000) -> Instance:
    """A small self-contained instance with no reference, used by the smoke run and the tests."""
    rng = np.random.default_rng(seed)
    cap = cap_for(num_secs, num_evs)
    shares = rng.dirichlet(np.full(num_secs, 4.0))
    petitions = np.round(shares * num_petitions).astype(int)
    petitions[0] += num_petitions - petitions.sum()
    mean = num_evs / num_secs
    values = np.zeros((num_secs, cap + 1), dtype=np.int64)
    energy = np.zeros((num_secs, cap + 1), dtype=np.float64)
    for s in range(num_secs):
        scale = max(0.5, rng.uniform(0.5, 1.5) * mean)
        values[s] = curve(int(petitions[s]), scale, cap)
        energy[s] = np.arange(cap + 1) * 100.0
    return Instance(
        name=name,
        source=f"synthetic, seed {seed}",
        num_evs=num_evs,
        num_petitions=num_petitions,
        positions=grid_positions(num_secs, city),
        connections=grid_connections(num_secs, connections if connections is not None else num_secs),
        values=values,
        energy=energy,
        notes="Dirichlet petition shares, saturating curves with a random scale around the mean allocation.",
    )
