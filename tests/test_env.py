"""Environment port: vehicle conservation, action semantics, episode bookkeeping."""
import numpy as np

from fleet_allocation_torch.env import EnergyEnvironment, TripsEnvironment
from fleet_allocation_torch.generator import synthetic


def make_env(rounds=3, seed=0):
    inst = synthetic("t", num_secs=6, num_evs=36, num_petitions=600, seed=seed)
    return inst, TripsEnvironment(inst, rounds=rounds, seed=seed)


def test_sizes_and_reset():
    inst, env = make_env()
    assert env.state_size == 9
    assert env.action_size == 2
    assert env.steps_per_episode == 3 * 6
    assert env.allocation.tolist() == inst.naive_allocation().tolist()
    env.state(0)
    env.step(1, 0)
    env.reset(seed=1)
    assert env.allocation.tolist() == inst.naive_allocation().tolist()
    assert env.steps == 0


def test_vehicles_are_conserved_and_serve_keeps_allocation():
    inst, env = make_env()
    for _ in range(env.rounds):
        for i in range(env.num_secs):
            before = env.allocation.copy()
            env.state(i)
            r, s2, done = env.step(0, i)
            assert env.allocation.tolist() == before.tolist()
            assert s2.shape == (9,)
            assert r >= 0
    assert done


def test_rebalance_never_lowers_the_pair_value_and_conserves_vehicles():
    inst, env = make_env(rounds=5)
    total = int(env.allocation.sum())
    for _ in range(env.rounds):
        for i in range(env.num_secs):
            env.state(i)
            p = int(env.partner[i])
            before = inst.value(i, env.allocation[i]) + inst.value(p, env.allocation[p])
            env.step(1, i)
            after = inst.value(i, env.allocation[i]) + inst.value(p, env.allocation[p])
            assert after >= before
            assert int(env.allocation.sum()) == total
            assert (env.allocation >= 1).all()
    assert env.total_trips_satisfied() >= inst.total_value(inst.naive_allocation())
    assert env.best_total >= env.total_trips_satisfied()


def test_partner_is_a_neighbour():
    inst, env = make_env()
    nb = inst.neighbours
    for i in range(env.num_secs):
        env.state(i)
        assert int(env.partner[i]) in nb[i] or not nb[i]


def test_observation_is_normalised_and_finite():
    inst, env = make_env()
    s = env.state(2)
    assert np.isfinite(s).all()
    assert s.dtype == np.float32
    assert 0 <= s[8] <= 1


def test_energy_reward_is_petitions_minus_energy():
    inst = synthetic("e", num_secs=4, num_evs=24, num_petitions=300, seed=0)
    env = EnergyEnvironment(inst, rounds=1, seed=0, alpha=0.5)
    env.state(0)
    r, _, _ = env.step(0, 0)
    p = int(env.partner[0])
    petitions = (inst.value(0, env.allocation[0]) + inst.value(p, env.allocation[p])) / 2
    assert r < petitions


def test_node_features_shape():
    inst, env = make_env()
    f = env.node_features()
    assert f.shape == (6, 4)
    assert np.isfinite(f).all()
