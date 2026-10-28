"""Benchmark instances for the vehicle-to-SEC allocation problem.

An instance holds the Smart Energy Communities (SECs) of a city, the undirected connections along which
two SECs may exchange autonomous electric vehicles (AEVs), the fleet size, the number of trip petitions and
one look-up table per SEC. The look-up table ``values[sec][j]`` is the number of trip petitions the SEC
satisfies when it operates ``j`` AEVs, and ``energy[sec][j]`` is the energy its fleet consumes. The tables
are precomputed by the ride-sharing simulator of the thesis (Chapter 3) or, for generated instances, by a
saturating curve fitted to a reference instance. The range of ``j`` is capped at four times the mean
allocation, where every thesis curve is flat, and a look-up beyond the cap returns the value at the cap.

Instances are stored as JSON under ``instances/`` so that they can be released on their own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = ROOT / "instances"


@dataclass
class Instance:
    """One allocation instance. SEC ids and connections are zero based."""

    name: str
    source: str
    num_evs: int
    num_petitions: int
    positions: list[tuple[int, int]]
    connections: list[tuple[int, int]]
    values: np.ndarray
    energy: np.ndarray
    notes: str = ""
    thesis: dict = field(default_factory=dict)

    @property
    def num_secs(self) -> int:
        return int(self.values.shape[0])

    @property
    def cap(self) -> int:
        return int(self.values.shape[1] - 1)

    @property
    def mean_allocation(self) -> int:
        return self.num_evs // self.num_secs

    @property
    def neighbours(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in range(self.num_secs)]
        for a, b in self.connections:
            out[a].append(b)
            out[b].append(a)
        return out

    def value(self, sec: int, j: int) -> int:
        """Trip petitions satisfied by ``sec`` with ``j`` AEVs, flat beyond the cap."""
        return int(self.values[sec, min(j, self.cap)])

    def energy_of(self, sec: int, j: int) -> float:
        return float(self.energy[sec, min(j, self.cap)])

    def total_value(self, allocation: np.ndarray) -> int:
        idx = np.minimum(allocation, self.cap)
        return int(self.values[np.arange(self.num_secs), idx].sum())

    def naive_allocation(self) -> np.ndarray:
        """Even split, the remainder going to the first SECs, as the thesis lower bound does."""
        base = self.num_evs // self.num_secs
        alloc = np.full(self.num_secs, base, dtype=np.int64)
        alloc[: self.num_evs - base * self.num_secs] += 1
        return alloc

    def upper_bound(self) -> int:
        """Sum of the per-SEC maxima, the bound the thesis MIP uses for the objective."""
        return int(self.values.max(axis=1).sum())

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "num_secs": self.num_secs,
            "num_evs": self.num_evs,
            "num_petitions": self.num_petitions,
            "cap": self.cap,
            "positions": [list(p) for p in self.positions],
            "connections": [list(c) for c in self.connections],
            "values": self.values.astype(int).tolist(),
            "energy": [[round(float(x), 1) for x in row] for row in self.energy],
            "notes": self.notes,
            "thesis": self.thesis,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Instance":
        return cls(
            name=data["name"],
            source=data["source"],
            num_evs=int(data["num_evs"]),
            num_petitions=int(data["num_petitions"]),
            positions=[tuple(p) for p in data["positions"]],
            connections=[tuple(c) for c in data["connections"]],
            values=np.asarray(data["values"], dtype=np.int64),
            energy=np.asarray(data["energy"], dtype=np.float64),
            notes=data.get("notes", ""),
            thesis=data.get("thesis", {}),
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or INSTANCES / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_json(), separators=(",", ":"))
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="\n")
        return path


def load_instance(name_or_path: str | Path) -> Instance:
    path = Path(name_or_path)
    if not path.exists():
        path = INSTANCES / f"{name_or_path}.json"
    return Instance.from_json(json.loads(path.read_text(encoding="utf-8")))


def cap_for(num_secs: int, num_evs: int) -> int:
    return 4 * (num_evs // num_secs)


def read_legacy_lookup(path: Path) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, float]]]:
    """Read a thesis ``subproblem_solutions.csv`` (path, trips satisfied, energy, separated by semicolons).

    The file is a simulator output from the thesis working tree, not raw trip data. SEC ids are one based
    in the file and are kept as they are here. The caller converts them.
    """
    values: dict[int, dict[int, int]] = {}
    energy: dict[int, dict[int, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p, tp, en = line.strip().split(";")
        stem = p.replace("\\", "/").split("/")[-1].split(".")[0]
        sec = int(stem.split("_")[1])
        j = int(stem.split("_num_EVs_")[1])
        values.setdefault(sec, {})[j] = int(tp)
        energy.setdefault(sec, {})[j] = float(en)
    return values, energy


def read_legacy_graph(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
    """Read the SEC positions, the connections, the AEV count and the petition count of a thesis ``input.in``.

    Only the header blocks are read. The vehicle schedules and the trip petitions themselves are skipped.
    """
    with path.open(encoding="utf-8") as f:
        f.readline()
        num_secs = int(f.readline().strip())
        positions: list[tuple[int, int]] = []
        for _ in range(num_secs):
            sec_id, x, y = map(int, f.readline().split())
            positions.append((x, y))
        num_conn = int(f.readline().strip())
        connections: list[tuple[int, int]] = []
        for _ in range(num_conn):
            a, b = map(int, f.readline().split())
            connections.append((min(a, b) - 1, max(a, b) - 1))
        num_evs = int(f.readline().strip())
        for _ in range(num_evs):
            f.readline()
            movs = int(f.readline().strip())
            for _ in range(movs):
                f.readline()
        num_tps = int(f.readline().strip())
    connections = sorted(set(connections))
    return positions, connections, num_evs, num_tps


def instance_from_legacy(name: str, source: str, lookup_csv: Path, input_in: Path, notes: str = "", thesis: dict | None = None) -> Instance:
    """Build a thesis-scale instance from the simulator look-up table and the instance header."""
    values_d, energy_d = read_legacy_lookup(lookup_csv)
    positions, connections, num_evs, num_tps = read_legacy_graph(input_in)
    num_secs = len(values_d)
    assert len(positions) == num_secs, f"{len(positions)} SEC positions for {num_secs} look-up tables"
    cap = cap_for(num_secs, num_evs)
    values = np.zeros((num_secs, cap + 1), dtype=np.int64)
    energy = np.zeros((num_secs, cap + 1), dtype=np.float64)
    for sec in range(num_secs):
        row = values_d[sec + 1]
        erow = energy_d[sec + 1]
        for j in range(1, cap + 1):
            values[sec, j] = row[j]
            energy[sec, j] = erow[j]
    return Instance(name=name, source=source, num_evs=num_evs, num_petitions=num_tps, positions=positions,
                    connections=connections, values=values, energy=energy, notes=notes, thesis=thesis or {})
