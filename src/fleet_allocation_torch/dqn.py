"""Deep Q-learning agent, ported from the thesis ``DeepTripsEnvironment`` (TensorFlow Keras) to PyTorch.

Reference configuration kept from the original: two hidden layers of 128 and 64 units with ReLU, a linear
output, mean squared error, Adam with learning rate 0.01, replay memory of 2,000 transitions, minibatch 64,
epsilon from 1.0 to 0.01 with decay 0.995 per replay, target network update every 200 steps, discount from
the constructor. The original fitted one transition at a time inside the minibatch loop. This port updates
on the whole minibatch in one optimiser step, see ``docs/port_notes.md``.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """128 and 64 hidden units with ReLU, linear output of one Q-value per action."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayMemory:
    """First in, first out memory of ``(state, action, reward, next_state, done)`` tuples."""

    def __init__(self, capacity: int = 2000, seed: int = 0) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, extra: float = 0.0) -> None:
        self.buffer.append((state, action, reward, next_state, done, extra))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(self.buffer, batch_size)
        states = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.int64)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.as_tensor([float(b[4]) for b in batch], dtype=torch.float32)
        extras = torch.as_tensor([b[5] for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones, extras


class DQNAgent:
    """Epsilon-greedy agent with a target network. ``model`` may be wrapped in DistributedDataParallel, in
    which case ``policy`` is the wrapped module and only ``replay`` runs the wrapped forward pass."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        model: nn.Module,
        target_model: nn.Module,
        learning_rate: float = 0.01,
        memory_size: int = 2000,
        batch_size: int = 64,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 200,
        seed: int = 0,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.policy = model.module if hasattr(model, "module") else model
        self.memory = ReplayMemory(memory_size, seed=seed)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.update_target()

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() <= self.epsilon:
            return int(self.rng.integers(self.action_size))
        with torch.no_grad():
            q = self.policy(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q[0]).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, shared: float = 0.0) -> None:
        self.memory.push(state, action, reward, next_state, done, shared)

    def replay(self) -> float | None:
        """One minibatch update. Returns the loss, or None when the memory is still too small."""
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones, shared = self.memory.sample(self.batch_size)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1).values
            target = rewards + shared + self.gamma * next_q * (1.0 - dones)
        q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return float(loss.item())

    def update_target(self) -> None:
        self.target_model.load_state_dict(self.policy.state_dict())
