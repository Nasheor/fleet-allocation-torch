"""GCN experience sharing on a 4-SEC ring with hand-computed adjacency and shared rewards."""
import numpy as np
import torch

from fleet_allocation_torch.gcn import GCN, ExperienceSharer, compute_adjacency_matrix, share_experience

RING = [[1, 3], [0, 2], [1, 3], [2, 0]]


def test_ring_adjacency_is_normalised_with_self_loops():
    adj = compute_adjacency_matrix(RING)
    third = 1.0 / 3.0
    expected = np.array([
        [third, third, 0.0, third],
        [third, third, third, 0.0],
        [0.0, third, third, third],
        [third, 0.0, third, third],
    ], dtype=np.float32)
    assert np.allclose(adj.numpy(), expected, atol=1e-6)
    assert np.allclose(adj.numpy(), adj.numpy().T)


def test_one_based_dict_input_matches_list_input():
    as_dict = {1: [2, 4], 2: [1, 3], 3: [2, 4], 4: [3, 1]}
    assert torch.equal(compute_adjacency_matrix(as_dict), compute_adjacency_matrix(RING))


def test_shared_rewards_on_the_ring_match_hand_computation():
    gcn = GCN(input_dim=2, hidden_dim=2, output_dim=1)
    with torch.no_grad():
        gcn.gc1.weight.copy_(torch.eye(2))
        gcn.gc1.bias.zero_()
        gcn.gc2.weight.copy_(torch.ones(1, 2))
        gcn.gc2.bias.zero_()
    gcn.eval()
    rewards = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    shared = share_experience(gcn, rewards, compute_adjacency_matrix(RING)).detach().numpy()
    # node i averages itself and its two ring neighbours: (1+2+4)/3, (1+2+3)/3, (2+3+4)/3, (3+4+1)/3
    expected = np.array([7 / 3, 2.0, 3.0, 8 / 3], dtype=np.float32)
    assert np.allclose(shared, expected, atol=1e-5)


def test_star_adjacency_degrees():
    star = [[1, 2, 3], [0], [0], [0]]
    adj = compute_adjacency_matrix(star).numpy()
    assert np.isclose(adj[0, 0], 1 / 4)
    assert np.isclose(adj[0, 1], 1 / np.sqrt(4 * 2))
    assert np.isclose(adj[1, 1], 1 / 2)
    assert adj[1, 2] == 0


def test_sharer_learns_and_scales_by_beta():
    torch.manual_seed(0)
    sharer = ExperienceSharer(GCN(4, 8, 1), compute_adjacency_matrix(RING), beta=0.5, reward_scale=10.0)
    feats = np.random.default_rng(0).random((4, 4)).astype(np.float32)
    rewards = np.array([10.0, 20.0, 30.0, 40.0])
    assert sharer.update(feats, rewards) is None
    first = sharer.update(feats, rewards)
    for _ in range(100):
        last = sharer.update(feats, rewards)
    assert first is not None and last is not None and last < first
    assert sharer.shared.shape == (4,)
