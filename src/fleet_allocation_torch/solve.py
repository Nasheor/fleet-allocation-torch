"""CP-SAT solves at thesis scale and at ten times the thesis scale, and the learned-policy comparison.

Results go to ``results/ortools_thesis_scale.json`` and ``results/ortools_x10.json``. Each record carries the
objective, the best bound, the gap, the wall seconds and the status of the allocation model and of the
negotiation-schedule colouring, and the x10 file adds the greedy policy's objective, wall time and gaps.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dqn import QNetwork
from .env import TripsEnvironment
from .instances import ROOT, Instance, load_instance
from .ortools_models import line_graph, solve_allocation, solve_colouring

RESULTS = ROOT / "results"


def solve_models(inst: Instance, time_limit: float, workers: int = 8) -> dict:
    alloc = solve_allocation(inst.values, inst.num_evs, ub=inst.upper_bound(), time_limit=time_limit, workers=workers)
    n, adjacency = line_graph(inst.num_secs, inst.connections)
    max_degree = max((len(a) for a in adjacency), default=0)
    col = solve_colouring(n, adjacency, time_limit=time_limit, workers=workers)
    return {
        "secs": inst.num_secs,
        "aevs": inst.num_evs,
        "petitions": inst.num_petitions,
        "cap": inst.cap,
        "connections": len(inst.connections),
        "naive_objective": int(inst.total_value(inst.naive_allocation())),
        "sum_of_maxima": int(inst.upper_bound()),
        "allocation": {
            "objective": alloc.objective,
            "bound": alloc.bound,
            "gap": round(alloc.gap, 6),
            "wall_s": round(alloc.wall_s, 2),
            "status": alloc.status,
            "allocated_aevs": int(sum(alloc.solution)) if alloc.solution else None,
        },
        "colouring": {
            "objective": col.objective,
            "bound": col.bound,
            "gap": round(col.gap, 6),
            "wall_s": round(col.wall_s, 2),
            "status": col.status,
            "nodes": n,
            "max_degree": max_degree,
            "lower_bound_from_degree": max_degree + 1 if n else 0,
        },
    }


def load_policy(path: Path) -> tuple[QNetwork, dict]:
    """Load a policy saved by ``train.py`` (``.pt``) or by ``save_policy_json`` (``.json``)."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        state = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data["state_dict"].items()}
        meta = {k: v for k, v in data.items() if k != "state_dict"}
    else:
        data = torch.load(path, map_location="cpu")
        state = data["state_dict"]
        meta = {k: v for k, v in data.items() if k != "state_dict"}
    net = QNetwork(int(meta["state_size"]), int(meta["action_size"]))
    net.load_state_dict(state)
    net.eval()
    return net, meta


def save_policy_json(pt_path: Path, out_path: Path) -> Path:
    data = torch.load(pt_path, map_location="cpu")
    out = {k: v for k, v in data.items() if k != "state_dict"}
    out["state_dict"] = {k: np.round(v.numpy(), 6).tolist() for k, v in data["state_dict"].items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8", newline="\n")
    return out_path


def greedy_episode(net: QNetwork, inst: Instance, rounds: int = 10, seed: int = 10_000) -> dict:
    """Run the greedy policy once from the naive allocation and report the trips satisfied and the wall time."""
    env = TripsEnvironment(inst, rounds=rounds, seed=seed)
    t0 = time.perf_counter()
    rebalances = 0
    with torch.no_grad():
        for _ in range(env.rounds):
            for i in range(env.num_secs):
                s = env.state(i)
                a = int(torch.argmax(net(torch.as_tensor(s).unsqueeze(0))[0]).item())
                rebalances += a
                env.step(a, i)
    wall = time.perf_counter() - t0
    return {
        "objective": int(env.total_trips_satisfied()),
        "best_seen": int(env.best_total),
        "wall_s": round(wall, 3),
        "rounds": rounds,
        "steps": env.steps_per_episode,
        "rebalance_actions": int(rebalances),
    }


def policy_comparison(inst: Instance, record: dict, policy_path: Path, rounds: int = 10) -> dict:
    net, meta = load_policy(policy_path)
    out = greedy_episode(net, inst, rounds=rounds)
    solver_obj = record["allocation"]["objective"]
    bound = record["allocation"]["bound"]
    out["policy_source"] = policy_path.name
    out["trained_on"] = meta.get("instance")
    out["trained_world_size"] = meta.get("world_size")
    out["naive_objective"] = record["naive_objective"]
    out["gap_to_solver"] = round((solver_obj - out["objective"]) / solver_obj, 6) if solver_obj else None
    out["gap_to_bound"] = round((bound - out["objective"]) / bound, 6) if bound else None
    return out


def run_scale(names: list[str], time_limit: float, out_file: Path, workers: int = 8, policy_path: Path | None = None,
              thesis: bool = False) -> dict:
    results: dict = {
        "time_limit_s": time_limit,
        "solver_workers": workers,
        "instances": {},
    }
    for name in names:
        inst = load_instance(name)
        rec = solve_models(inst, time_limit, workers)
        if thesis and inst.thesis:
            rec["thesis"] = {
                **inst.thesis,
                "naive_from_tables": rec["naive_objective"],
                "naive_matches_qr_lb": rec["naive_objective"] == inst.thesis.get("qr_lb"),
                "cp_sat_matches_qr_ub": rec["allocation"]["objective"] == inst.thesis.get("qr_ub"),
            }
        if policy_path is not None and policy_path.exists():
            rec["policy"] = policy_comparison(inst, rec, policy_path)
        results["instances"][name] = rec
        print(json.dumps({name: {k: v for k, v in rec.items() if k in ("allocation", "colouring", "policy", "thesis")}}), flush=True)
    results["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8", newline="\n")
    return results


def thesis_scale(time_limit: float = 600.0, policy_path: Path | None = None) -> dict:
    return run_scale(["GHC_ai", "NYC_ai"], time_limit, RESULTS / "ortools_thesis_scale.json", policy_path=policy_path, thesis=True)


def x10_scale(time_limit: float = 1800.0, policy_path: Path | None = None) -> dict:
    return run_scale(["GHC_x10", "NYC_x10"], time_limit, RESULTS / "ortools_x10.json", policy_path=policy_path)
