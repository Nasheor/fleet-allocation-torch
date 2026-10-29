# fleet-allocation-torch

The vehicle-to-community allocation chapter of the PhD thesis *A Framework for Adaptive Ride-Sharing and
Management of Electric Vehicles* (Munster Technological University, 2025), made reproducible. The thesis
trained one Deep Q-Learning agent per Smart Energy Community (SEC) in TensorFlow Keras, left a Graph
Convolutional Network for experience sharing as a stub, and solved the centralised allocation and the
negotiation schedule with CPLEX. This repo ports the training to PyTorch with `torch.distributed`
DistributedDataParallel over gloo on CPU workers, completes the GCN, measures how the training scales from
one to four workers, ports the two optimisation models to OR-Tools CP-SAT, re-solves them at the thesis
scale and at ten times that scale, and compares the learned policy with the solver on the same instances.
Every number in the results section comes from `results/*.json` and is asserted by the tests.

## Publication status

This repository stays unpublished until the history of the original thesis repositories on GitHub has been
purged and the personal access token that sits in that history has been revoked. The port was built from
the working tree of the original code only, never from its git history, and `tests/test_no_secrets.py`
passes on every file here, including `legacy_reference/`. The clean-up steps for the author's own account
are in `tools/purge_notes.md`. Until they are done the repo is not pushed anywhere.

<!-- results:start -->
## Results

Not generated yet.
<!-- results:end -->

## What was ported

| Original (thesis code) | This repo |
|---|---|
| `Q_Environment/TripsEnvironment.py`, `EnergyEnvironment.py` (tabular Q-learning environments with a community agent) | `env/trips_env.py`, `env/energy_env.py`, plain numpy with `reset(seed)`, `step(action, community_index)`, `state_size`, `action_size` |
| `Deep_Q_Environment/DeepTripsEnvironment.py` (Keras DQN, one sample at a time) | `dqn.py`: `QNetwork` (128, 64, linear), `ReplayMemory`, `DQNAgent` with `act`, `remember`, `replay`, `update_target`, one minibatch update of 64 per step |
| `Deep_Q_Environment/GCNExperienceSharing.py` (GCN module, two stubs) | `gcn.py`: the same `GCN`, `compute_adjacency_matrix` (symmetric normalised adjacency with self loops), `share_experience`, and an online trainer for the shared reward |
| `ride_sharing_framework/3_Code/.../mip_models.py` (docplex) | `ortools_models.py`: the same two functions with the same signatures on CP-SAT, plus array-based versions that return bound, gap, wall time and status |
| `ride_sharing_framework/1_Instance_Generator/` (grid of SECs, connection rule, trip data) | `generator.py`: the grid and the connection rule, with per-SEC look-up curves resampled from a reference instance |
| none | `train.py` (torchrun, DDP), `scaling.py`, `solve.py`, `report.py`, `instances.py`, `build_instances.py` |

`docs/port_notes.md` lists every behavioural difference, including the batched update, the local
observation, the honoured `serve` action, the episode reset and the energy reward sign.

## Method

**Problem.** A city has `numC` SECs connected by an undirected graph and a fleet of `numE` identical
autonomous electric vehicles (AEVs). Each SEC holds a look-up table giving the trip petitions it satisfies
with `j` vehicles, precomputed by the thesis ride-sharing simulator. The allocation that maximises the total
is a multiple-choice knapsack (the thesis's centralised MIP). The thesis's decentralised alternative lets
pairs of connected SECs negotiate vehicle exchanges from local information, with a DQN agent per SEC.

**Instances.** `GHC_ai` (16 SECs, 400 AEVs, 10,000 petitions, from the Google Hash Code 2018
`d_metropolis.in` alignment) and `NYC_ai` (256 SECs, 1,536 AEVs, 50,000 petitions, from the 2018 NYC green
taxi alignment) carry the look-up tables the thesis simulator produced in January 2024. The even allocation of
those tables gives 4,679 and 34,091 trips, which are the naive bounds printed on page 83 of the thesis, so
the tables are the thesis's own. `GHC_x10` (160 SECs, 4,000 AEVs, 100,000 petitions) and `NYC_x10` (2,560
SECs, 15,360 AEVs, 500,000 petitions) are generated with the thesis's grid and connection rule and per-SEC
saturating curves resampled from the thesis-scale tables, because running the simulator for 2,560 SECs
times 24 fleet sizes would take far more than one session. The look-up range of every instance is capped at
four times the mean allocation, where every thesis curve is already flat (the GHC curves saturate by 55
vehicles against a cap of 100, the NYC curves by 13 against a cap of 24).

**Environment.** On its turn a SEC is paired with a connected neighbour by the original rule and chooses
`serve` (keep the allocation) or `rebalance` (pool the pair's vehicles and split them by the tables). The
reward is the pair's mean petitions satisfied after the action, as in the original. An episode is ten passes
over the SECs from the even allocation, and the greedy evaluation reports the trips satisfied at the end of
one such episode. The observation is the acting SEC's and its partner's vehicles, trips, marginal gain and
marginal loss, normalised, plus the SEC's degree.

**Training.** `torchrun --nproc_per_node=N -m fleet_allocation_torch.train --instance GHC_ai --episodes E --seed S`.
Each rank owns an environment seeded by `seed + rank` and its own replay memory, the Q-network is wrapped in
DistributedDataParallel, every rank updates on its own minibatch of 64 with the gradients averaged inside
the backward pass, the target network is broadcast from rank 0 every 200 steps, and rank 0 evaluates the
greedy policy every ten episodes. The GCN takes the per-SEC features (vehicles, trips, energy, last reward),
is trained once per pass to predict each SEC's next reward from its neighbourhood, and its detached
prediction times 0.1 is added to the Q target of that SEC's transitions as the shared reward.

**Scaling table.** The same number of episodes per rank with 1, 2 and 4 workers, one torch thread per
worker, seed 0. Environment steps per second and updates per second are summed over ranks. Time to target
is the wall time at which the evaluation first reaches 90 percent of the single-worker final reward, the
final reward being the mean of the last ten evaluations.

**Solver.** CP-SAT with 8 workers. Allocation: one boolean per SEC and vehicle count, exactly one count per
SEC, at most the fleet size in total, maximise the summed table values with the sum of the per-SEC maxima
as the objective bound. Colouring: one node per SEC connection, two nodes adjacent when the connections
share a SEC, one colour per node, adjacent nodes differ, minimise the number of colours, with the original
symmetry breaking. Tests check both models against brute force on tiny instances.

**Policy comparison.** The policy from the largest worker count that trained on `GHC_ai` is run greedily
once on each ten-times instance from the even allocation. Its objective is compared with the solver's
objective and with the solver's best bound.

## How to run

```
C:/Users/nagar/anaconda3/python.exe -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e .[dev]
set PYTHONPATH=src
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m fleet_allocation_torch smoke
.venv\Scripts\python -m fleet_allocation_torch scaling --episodes 300
.venv\Scripts\python -m fleet_allocation_torch solve-thesis --time-limit 600
.venv\Scripts\python -m fleet_allocation_torch solve-x10 --time-limit 1800
.venv\Scripts\python -m fleet_allocation_torch parity
```

The smoke run trains on `tiny_4` with 1 and 2 workers for 20 episodes each, solves `tiny_4` and `smoke_40`
with CP-SAT, and refreshes the tables above in under ten minutes. `torchrun` on Windows needs
`USE_LIBUV=0` in the environment, which `scaling.py` sets for the processes it starts. No key is needed and
no paid call is made. `costcap.py` is present from the workspace template and the spend ledger records zero.

## Limits

- CPU only, gloo backend, one machine. No NCCL, no GPU, no cloud workers.
- One seed for the scaling table. Throughput is stable across seeds, time to target is not, and the table
  should be read as one run.
- The look-up tables at thesis scale are the thesis simulator's output. At ten times the scale they are
  fitted curves, not simulator runs, so the ten-times objectives describe a generated instance with the
  thesis's per-SEC distributions, not a new real-data alignment.
- The agent has the original's two actions. `rebalance` never lowers the pair's petitions, so the policy to
  learn is when to rebalance with the partner the original rule chose, and the comparison with the solver
  measures how far pairwise negotiation along the graph gets from the even allocation, not the strength of
  the network.
- The legacy TensorFlow curve was not reproduced on this machine (no TensorFlow environment). The port is
  compared with the thesis's printed figures instead.
- CP-SAT time limits of 600 s and 1,800 s. Where the status is `FEASIBLE` the gap to the bound is reported.

## Instance release status

| Source | Terms | Instance files released |
|---|---|---|
| Google Hash Code 2018 qualification input `d_metropolis.in` | Apache License 2.0 (Google's competition archive on GitHub) | `GHC_ai.json`, `GHC_x10.json` |
| NYC TLC 2018 green taxi trip records and NYC Taxi Zones | NYC Open Data Law: no licence requirement or usage restriction, source and modifications to be identified, the City may ask for distribution to stop | `NYC_ai.json`, `NYC_x10.json` |
| This repo's generator | MIT | `tiny_4.json`, `smoke_40.json` |

`LICENCE_CHECK.md` holds the rows, `instances/README.md` the source lines, and `tests/test_instances.py`
checks that every instance file is covered by an allowing row. No trip row, ride row or map file from either
source is in the repo, only SEC positions, SEC connections and per-SEC counts.

## Data and licences

See `LICENCE_CHECK.md`. No personal data is used. The original thesis code in `legacy_reference/` is the
author's own and keeps the Apache 2.0 licence file it came with. The port and everything else here is MIT.

## Licence

MIT, Avinash Nagarajan.
