"""Energy environment, ported from the thesis ``EnergyEnvironment`` and ``DeepEnergyEnvironment``.

The dynamics are those of the trips environment. Two things differ, as in the original: a SEC prefers a
partner whose fleet consumes less energy than its own, and the reward weighs energy against petitions. The
original code rewarded the mean energy consumed with a positive sign, which a maximising learner would push
up and which contradicts the thesis text (reward = petitions served minus alpha times energy). The port
follows the thesis text. See ``docs/port_notes.md``.
"""
from __future__ import annotations

import numpy as np

from ..instances import Instance
from .trips_env import TripsEnvironment


class EnergyEnvironment(TripsEnvironment):
    def __init__(self, instance: Instance, rounds: int = 10, seed: int = 0, alpha: float = 0.5) -> None:
        super().__init__(instance, rounds=rounds, seed=seed)
        self.alpha = alpha
        self.energy_per_ev = max(float(np.median(instance.energy[:, 1])), 1.0)

    def energy(self, sec: int, j: int) -> float:
        return self.instance.energy_of(sec, int(j))

    def score(self, sec: int) -> float:
        """Negative energy, so that the partner rule of the trips environment prefers lower consumption."""
        return -self.energy(sec, self.allocation[sec])

    def reward(self, sec: int, partner: int) -> float:
        petitions = (self.value(sec, self.allocation[sec]) + self.value(partner, self.allocation[partner])) / 2.0
        energy = (self.energy(sec, self.allocation[sec]) + self.energy(partner, self.allocation[partner])) / 2.0
        return petitions - self.alpha * energy / self.energy_per_ev
