"""DQN agent: network shape, replay gating, batched update, target copy."""
import numpy as np
import torch

from fleet_allocation_torch.dqn import DQNAgent, QNetwork, ReplayMemory


def test_qnetwork_layers_and_shape():
    net = QNetwork(9, 2)
    sizes = [m.out_features for m in net.net if isinstance(m, torch.nn.Linear)]
    assert sizes == [128, 64, 2]
    assert net(torch.zeros(5, 9)).shape == (5, 2)


def test_replay_waits_for_a_full_batch_then_updates_once_per_call():
    torch.manual_seed(0)
    agent = DQNAgent(9, 2, gamma=0.8, model=QNetwork(9, 2), target_model=QNetwork(9, 2), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(63):
        s = rng.random(9).astype(np.float32)
        agent.remember(s, int(rng.integers(2)), float(rng.random()), s, False)
    assert agent.replay() is None
    assert agent.updates == 0
    agent.remember(rng.random(9).astype(np.float32), 0, 1.0, rng.random(9).astype(np.float32), True)
    eps = agent.epsilon
    loss = agent.replay()
    assert loss is not None and loss >= 0
    assert agent.updates == 1
    assert agent.epsilon < eps
    assert len(agent.memory) == 64


def test_memory_is_fifo_with_capacity():
    m = ReplayMemory(capacity=3, seed=0)
    for k in range(5):
        m.push(np.zeros(2), k, 0.0, np.zeros(2), False)
    assert len(m) == 3
    assert [b[1] for b in m.buffer] == [2, 3, 4]


def test_update_target_copies_weights_and_greedy_act_is_argmax():
    torch.manual_seed(1)
    model, target = QNetwork(9, 2), QNetwork(9, 2)
    agent = DQNAgent(9, 2, gamma=0.8, model=model, target_model=target, seed=0)
    for a, b in zip(model.state_dict().values(), target.state_dict().values()):
        assert torch.equal(a, b)
    s = np.ones(9, dtype=np.float32)
    with torch.no_grad():
        q = model(torch.as_tensor(s).unsqueeze(0))[0]
    assert agent.act(s, greedy=True) == int(torch.argmax(q))
