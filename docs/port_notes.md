# Port notes

Every behavioural difference between the original thesis code (`legacy_reference/`, TensorFlow Keras and
docplex) and the PyTorch and OR-Tools port in `src/fleet_allocation_torch/`. The original files are kept
verbatim for comparison and are not run.

## Files not copied

- The original repository README was not copied into `legacy_reference/`. The working-tree copy fails
  `tests/test_no_secrets.py`, so it stays out of this repo and `tools/purge_notes.md` describes the clean-up
  of the old repositories for Avinash. The file name is recorded here, its content is not.
- `instance_format.txt`, `taxi_zones_ny.txt` (NYC taxi zone geometry) and the `utilities/` folder were not on
  the list of files to copy and are not needed by the port.

## Learning agent (`dqn.py`)

1. **Batched update.** The original `replay` looped over the 64 sampled transitions and called
   `model.predict` and `model.fit` once per transition, so one replay was 64 single-sample gradient steps
   plus 128 single-sample forward passes. The port samples the same minibatch of 64 and performs one
   optimiser step on the whole batch. With Adam at the original learning rate this is the standard
   minibatch update, and it is what makes DistributedDataParallel meaningful (one gradient all-reduce per
   minibatch instead of 64).
2. **Loss on the chosen action.** The original built a full target vector by copying the network's own
   prediction and overwriting the chosen action's entry, then fitted the vector with mean squared error.
   Every entry other than the chosen action had zero error, so the original loss equalled the squared error
   of the chosen action divided by the number of actions. The port computes the mean squared error of the
   chosen action's Q-value directly. The two losses differ by a constant factor, which Adam absorbs.
3. **Hyperparameters kept.** Two hidden layers of 128 and 64 units with ReLU, linear output, mean squared
   error, Adam with learning rate 0.01, replay memory of 2,000 transitions, minibatch 64, epsilon from 1.0 to
   0.01 with decay 0.995 per replay, target network update every 200 environment steps, discount 0.8 (the
   value `main.py` passed in the original).
4. **Seeds.** Each worker seeds its environment, its replay sampler and its epsilon-greedy draws with
   `seed + rank`. The network initialisation uses `seed` and is broadcast from rank 0 under DDP.

## Environment (`env/trips_env.py`, `env/energy_env.py`)

5. **Observation.** The original fed the network the state of every community (identifier, available
   vehicles, trips satisfied, energy consumed, four raw numbers each) and gave no indication of which
   community was acting, so the same input could require different actions. The port feeds the acting
   SEC's own state and its partner's state, nine normalised features (vehicles, trips satisfied, marginal
   gain of one more vehicle, marginal loss of one fewer vehicle, for both SECs, plus the SEC's degree).
   This is the local information the thesis says each SEC negotiates with, and it lets one policy run on
   any instance size, which the ten-times-scale comparison relies on.
6. **The action is honoured.** The original `step` ignored the chosen action and always rebalanced (the
   `serve` branch was commented out). The port keeps the two actions of the original, `serve` leaves the
   allocation untouched and `rebalance` pools the pair's vehicles and splits them by the look-up tables.
7. **Rebalance split.** The original tried every split with the acting side below 50 vehicles and the
   partner above 0. The port tries every split with at least one vehicle on each side. The hard limit of 50
   is gone because the look-up range is capped at four times the mean allocation and is flat beyond it,
   which is above 50 for the GHC instances and far above the saturation point of every NYC curve.
8. **Reset between episodes.** The original never reset the environment between episodes (the call was
   commented out), so later episodes continued from the previous allocation, and the best configuration
   seen was tracked across the whole run. The port resets to the naive even allocation at the start of
   every episode, tracks the best configuration within the episode, and evaluates a greedy episode from the
   naive allocation, which is the thesis's starting point for the negotiation.
9. **Episode length.** `DeepTripsEnvironment` made one pass over the communities per episode and
   `DeepEnergyEnvironment` made `num_time_steps` passes. The port uses `rounds` passes for both, default
   10, so that an episode holds enough negotiations to move vehicles across the graph.
10. **Partner choice kept.** The partner is a random neighbour, replaced with probability 0.9 by the last
    neighbour in the list whose score is higher than the SEC's own, as in the original. The trips
    environment scores by trips satisfied and the energy environment by lower energy, as the originals did.
11. **Energy reward sign.** The original energy environment rewarded the mean energy consumed with a
    positive sign, which a maximising learner would push up, and which contradicts the thesis text (reward
    equals petitions served minus alpha times energy). The port follows the thesis text with alpha 0.5 and
    energy normalised by the energy per vehicle.
12. **Look-up table at zero vehicles.** The original tables start at one vehicle. The port adds the entry
    for zero vehicles with zero trips so that the solver and the environment index the same array.
13. **Vehicles are conserved.** The port asserts in its tests that the fleet size never changes and that no
    SEC ends with fewer than one vehicle, which the original relied on but did not check.

## Experience sharing (`gcn.py`)

14. **Stubs completed.** In the original, `compute_adjacency_matrix` returned an ellipsis and
    `run_experience_sharing` called `agent.get_experience`, `agent.reward` and `agent.update_q_value`, which
    no environment class defined. The port builds the symmetric normalised adjacency with self loops from
    the neighbours of each SEC, and `share_experience` passes per-SEC features (vehicles, trips satisfied,
    energy, last reward) through the two-layer GCN of the original, aggregating with the adjacency after the
    second layer as the original did.
15. **Training signal.** The original gave the GCN no loss. The port trains it online to predict each SEC's
    next-round reward from the neighbourhood features (Adam, learning rate 0.001 as in the thesis table),
    and the shared reward handed to the agents is the detached prediction scaled by beta (default 0.1),
    added to the Q target of that SEC's transitions as an additive term. Setting `--gcn-beta 0` disables
    it. The GCN is trained per worker and is not wrapped in DistributedDataParallel.

## Optimisation models (`ortools_models.py`)

16. **CP-SAT instead of docplex.** The allocation model keeps the binary variables, the one-count-per-SEC
    constraint, the objective bound `ub_TPs` and the maximisation. The original constrained the allocated
    total to equal the fleet size. The port uses "at most" the fleet size because the tables never decrease
    with the vehicle count, so the optimum is the same, and the model stays feasible when the per-SEC
    range is capped. The function with the original name and signature uses the equality so that it
    returns the original's values.
17. **Colouring.** The original coloured the nodes with colours in `[0, best_value - 2]` and minimised a
    variable bounded below by every colour plus one, with symmetry breaking on the first node that has a
    neighbour and colour 0 for isolated nodes. The port keeps all of it and exposes the status, bound and
    gap of the solver as well.
18. **Status.** The original reported `succeed` as 1 for optimal, 0 for a solution without proof and -1
    for none. The port's original-signature functions return the same, the array functions return the
    CP-SAT status name.

## Distributed training (`train.py`)

19. Each rank owns one environment and one replay memory. The Q-network is wrapped in
    DistributedDataParallel over gloo, the gradient all-reduce happens inside `loss.backward()` of every
    minibatch update, the target network is copied on every rank and then broadcast from rank 0 every 200
    steps, and rank 0 alone evaluates the greedy policy on a fixed evaluation environment. The evaluation
    forward pass runs on the unwrapped module so that it needs no collective call.
20. Each torch process uses one thread, so the scaling table measures data-parallel workers rather than
    intra-op threads.
