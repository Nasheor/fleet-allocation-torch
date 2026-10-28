"""Trips environment, ported from the thesis ``TripsEnvironment`` and ``DeepTripsEnvironment``.

Each SEC is an agent. On its turn a SEC is paired with one connected neighbour and chooses between two
actions: ``serve`` keeps the current allocation, ``rebalance`` pools the AEVs of the pair and splits them so
that the two look-up tables add up to the most trip petitions. The reward is the mean of the two SECs'
petitions satisfied after the action, as in the original. See ``docs/port_notes.md`` for every difference
from the original code.
"""
from __future__ import annotations

import numpy as np

from ..instances import Instance

ACTIONS = ("serve", "rebalance")


class TripsEnvironment:
    """One episode is ``rounds`` passes over the SECs. ``step`` acts for one SEC and returns
    ``(reward, next_state, done)`` like the original."""

    STATE_SIZE = 9

    def __init__(self, instance: Instance, rounds: int = 10, seed: int = 0) -> None:
        self.instance = instance
        self.rounds = rounds
        self.neighbours = instance.neighbours
        self.ev_scale = max(instance.num_evs / instance.num_secs, 1.0)
        self.tp_scale = max(instance.num_petitions / instance.num_secs, 1.0)
        self.rng = np.random.default_rng(seed)
        self.allocation = instance.naive_allocation()
        self.partner = np.zeros(instance.num_secs, dtype=np.int64)
        self.last_reward = np.zeros(instance.num_secs, dtype=np.float64)
        self.steps = 0
        self.best_total = self.total_trips_satisfied()
        self.best_allocation = self.allocation.copy()

    # ---- properties the training loop reads
    @property
    def state_size(self) -> int:
        return self.STATE_SIZE

    @property
    def action_size(self) -> int:
        return len(ACTIONS)

    @property
    def num_secs(self) -> int:
        return self.instance.num_secs

    @property
    def steps_per_episode(self) -> int:
        return self.rounds * self.instance.num_secs

    # ---- episode control
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.allocation = self.instance.naive_allocation()
        self.last_reward[:] = 0.0
        self.steps = 0
        self.best_total = self.total_trips_satisfied()
        self.best_allocation = self.allocation.copy()

    def total_trips_satisfied(self) -> int:
        return self.instance.total_value(self.allocation)

    def total_energy(self) -> float:
        idx = np.minimum(self.allocation, self.instance.cap)
        return float(self.instance.energy[np.arange(self.num_secs), idx].sum())

    # ---- look-ups
    def value(self, sec: int, j: int) -> int:
        return self.instance.value(sec, int(j))

    def score(self, sec: int) -> float:
        """The quantity a SEC compares with its neighbours when it picks a partner."""
        return float(self.value(sec, self.allocation[sec]))

    def choose_partner(self, sec: int) -> int:
        """The original rule: a random neighbour, replaced with probability 0.9 by the last neighbour in the
        list whose score is above the SEC's own score."""
        nb = self.neighbours[sec]
        if not nb:
            return sec
        partner = int(nb[self.rng.integers(len(nb))])
        if self.rng.uniform(0.0, 1.0) > 0.1:
            own = self.score(sec)
            for n in nb:
                if (self.score(n) + own) / 2.0 > own:
                    partner = int(n)
        return partner

    def observe(self, sec: int, partner: int) -> np.ndarray:
        """Local observation of the acting SEC and its partner, normalised by the instance scales."""
        a, b = int(self.allocation[sec]), int(self.allocation[partner])
        va, vb = self.value(sec, a), self.value(partner, b)
        return np.array(
            [
                a / self.ev_scale,
                va / self.tp_scale,
                (self.value(sec, a + 1) - va) / self.tp_scale,
                (va - self.value(sec, max(a - 1, 0))) / self.tp_scale,
                b / self.ev_scale,
                vb / self.tp_scale,
                (self.value(partner, b + 1) - vb) / self.tp_scale,
                (vb - self.value(partner, max(b - 1, 0))) / self.tp_scale,
                len(self.neighbours[sec]) / 8.0,
            ],
            dtype=np.float32,
        )

    def state(self, sec: int) -> np.ndarray:
        """Pick the partner for ``sec`` and return the observation. Call once before ``step``."""
        self.partner[sec] = self.choose_partner(sec)
        return self.observe(sec, int(self.partner[sec]))

    # ---- dynamics
    def best_split(self, sec: int, partner: int) -> tuple[int, int]:
        """Split the pooled AEVs of the pair to maximise the summed look-up values, the first best split
        winning ties. Both sides keep at least one AEV, as in the original."""
        pool = int(self.allocation[sec] + self.allocation[partner])
        if pool < 2:
            return int(self.allocation[sec]), int(self.allocation[partner])
        best, best_v = (int(self.allocation[sec]), int(self.allocation[partner])), -1
        for v in range(1, pool):
            total = self.value(sec, v) + self.value(partner, pool - v)
            if total > best_v:
                best_v, best = total, (v, pool - v)
        return best

    def rebalance(self, sec: int, partner: int) -> None:
        if partner == sec or self.allocation[sec] < 2:
            return
        a, b = self.best_split(sec, partner)
        self.allocation[sec], self.allocation[partner] = a, b

    def reward(self, sec: int, partner: int) -> float:
        return (self.value(sec, self.allocation[sec]) + self.value(partner, self.allocation[partner])) / 2.0

    def step(self, action: int, community_index: int) -> tuple[float, np.ndarray, bool]:
        sec = int(community_index)
        partner = int(self.partner[sec])
        if ACTIONS[action] == "rebalance":
            self.rebalance(sec, partner)
        r = self.reward(sec, partner)
        self.last_reward[sec] = r
        self.steps += 1
        total = self.total_trips_satisfied()
        if total > self.best_total:
            self.best_total = total
            self.best_allocation = self.allocation.copy()
        done = self.steps >= self.steps_per_episode
        return r, self.observe(sec, partner), done

    # ---- features for the experience-sharing graph network
    def node_features(self) -> np.ndarray:
        """Per-SEC features the thesis names for the graph network: AEVs, trips satisfied, energy, last reward."""
        idx = np.minimum(self.allocation, self.instance.cap)
        trips = self.instance.values[np.arange(self.num_secs), idx]
        energy = self.instance.energy[np.arange(self.num_secs), idx]
        e_scale = max(float(self.instance.energy[:, 1].mean()) * self.ev_scale, 1.0)
        return np.stack(
            [self.allocation / self.ev_scale, trips / self.tp_scale, energy / e_scale, self.last_reward / self.tp_scale],
            axis=1,
        ).astype(np.float32)
