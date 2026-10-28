"""Distributed training of the DQN agent with torch.distributed DistributedDataParallel over gloo.

Run with torchrun, for example::

    python -m torch.distributed.run --nproc_per_node=4 -m fleet_allocation_torch.train --instance GHC_ai --episodes 100 --seed 0

Each rank owns one environment copy seeded by ``seed + rank``, the Q-network is wrapped in
DistributedDataParallel so that the gradient all-reduce happens inside the backward pass of every minibatch
update, the target network is broadcast from rank 0 every 200 steps, and rank 0 evaluates the greedy policy
on a fixed evaluation environment every ``--eval-every`` episodes, writing one JSON line per evaluation to
``runs/<instance>_w<world>_s<seed>.jsonl``. Without torchrun the script runs as a single worker.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .dqn import DQNAgent, QNetwork
from .env import EnergyEnvironment, TripsEnvironment
from .gcn import GCN, ExperienceSharer, compute_adjacency_matrix
from .instances import load_instance

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fleet_allocation_torch.train")
    p.add_argument("--instance", required=True, help="instance name under instances/ or a path to a JSON file")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=10, help="passes over the SECs per episode")
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--gcn-beta", type=float, default=0.1, help="weight of the shared reward in the Q target, 0 disables it")
    p.add_argument("--env", choices=["trips", "energy"], default="trips")
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--out", default=str(RUNS))
    p.add_argument("--tag", default=None, help="run name, default <instance>_w<world>_s<seed>")
    p.add_argument("--save-policy", action="store_true", help="rank 0 saves the policy state dict next to the log")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def init_distributed() -> tuple[int, int]:
    """Join the process group when launched by torchrun, otherwise run alone."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def broadcast_module(module: torch.nn.Module, src: int = 0) -> None:
    for t in module.state_dict().values():
        dist.broadcast(t, src=src)


def evaluate(agent: DQNAgent, env: TripsEnvironment, seed: int) -> dict:
    """One greedy episode on a freshly reset environment. Returns the trips satisfied at the end, the best
    seen during the episode and the summed reward."""
    env.reset(seed=seed)
    ret = 0.0
    for _ in range(env.rounds):
        for i in range(env.num_secs):
            s = env.state(i)
            a = agent.act(s, greedy=True)
            r, _, _ = env.step(a, i)
            ret += r
    return {"eval_trips": int(env.total_trips_satisfied()), "eval_best_trips": int(env.best_total), "eval_return": round(ret, 3)}


def run(args: argparse.Namespace) -> dict:
    rank, world = init_distributed()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    instance = load_instance(args.instance)
    env_cls = EnergyEnvironment if args.env == "energy" else TripsEnvironment
    env = env_cls(instance, rounds=args.rounds, seed=args.seed + rank)
    eval_env = env_cls(instance, rounds=args.rounds, seed=10_000 + args.seed)

    model = QNetwork(env.state_size, env.action_size)
    target = QNetwork(env.state_size, env.action_size)
    if world > 1:
        broadcast_module(model)
        model = DDP(model)
    agent = DQNAgent(env.state_size, env.action_size, gamma=args.gamma, model=model, target_model=target, seed=args.seed + rank)

    sharer = None
    if args.gcn_beta > 0:
        gcn = GCN(input_dim=4, hidden_dim=16, output_dim=1)
        sharer = ExperienceSharer(gcn, compute_adjacency_matrix(env.neighbours), beta=args.gcn_beta, reward_scale=env.tp_scale)

    tag = args.tag or f"{instance.name}_w{world}_s{args.seed}"
    out_dir = Path(args.out)
    log_path = out_dir / f"{tag}.jsonl"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    step_count = 0
    last_loss = None
    last_eval: dict = {}
    for episode in range(args.episodes):
        env.reset()
        for _ in range(env.rounds):
            for i in range(env.num_secs):
                s = env.state(i)
                a = agent.act(s)
                r, s2, done = env.step(a, i)
                shared = float(sharer.shared[i]) if sharer is not None else 0.0
                agent.remember(s, a, r, s2, done, shared=shared)
                loss = agent.replay()
                if loss is not None:
                    last_loss = loss
                step_count += 1
                if step_count % agent.target_update_freq == 0:
                    agent.update_target()
                    if world > 1:
                        broadcast_module(agent.target_model)
            if sharer is not None:
                sharer.update(env.node_features(), env.last_reward)
        if rank == 0 and (episode + 1) % args.eval_every == 0:
            last_eval = evaluate(agent, eval_env, seed=10_000 + args.seed)
            row = {
                "episode": episode + 1,
                "wall_s": round(time.perf_counter() - t0, 3),
                "env_steps_total": step_count * world,
                "updates_total": agent.updates * world,
                "epsilon": round(agent.epsilon, 4),
                "loss": None if last_loss is None else round(last_loss, 4),
                "gcn_loss": None if sharer is None or not sharer.losses else round(sharer.losses[-1], 6),
                **last_eval,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if not args.quiet:
                print(json.dumps(row), flush=True)
    wall = time.perf_counter() - t0
    summary = {
        "final": True,
        "tag": tag,
        "instance": instance.name,
        "world_size": world,
        "seed": args.seed,
        "episodes": args.episodes,
        "rounds": args.rounds,
        "threads": args.threads,
        "wall_s": round(wall, 3),
        "env_steps_total": step_count * world,
        "updates_total": agent.updates * world,
        "env_steps_per_s": round(step_count * world / wall, 2),
        "updates_per_s": round(agent.updates * world / wall, 2),
        "naive_trips": int(instance.total_value(instance.naive_allocation())),
        "upper_bound": int(instance.upper_bound()),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **last_eval,
    }
    if rank == 0:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        if args.save_policy:
            torch.save({"state_dict": agent.policy.state_dict(), "state_size": env.state_size, "action_size": env.action_size,
                        "instance": instance.name, "world_size": world, "seed": args.seed, "episodes": args.episodes},
                       out_dir / f"{tag}.pt")
        if not args.quiet:
            print(json.dumps(summary), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
