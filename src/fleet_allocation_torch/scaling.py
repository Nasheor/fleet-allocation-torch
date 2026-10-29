"""Scaling runs: the same training on 1, 2 and 4 gloo workers, and the scaling table.

The workers are started by torchrun. When torchrun cannot create its rendezvous store on this machine (the
Windows CPU wheel of PyTorch is built without libuv and the torchrun agent of torch 2.15 asks for libuv
regardless of ``USE_LIBUV``), the launcher records the error and starts the same number of ranks itself with
the ``env://`` rendezvous (``MASTER_ADDR``, ``MASTER_PORT``, ``RANK``, ``WORLD_SIZE``), which is the
rendezvous torchrun hands its workers. The training script, the process group and the DistributedDataParallel
wrapper are the same either way, and the result file names the launcher used.

Metrics per worker count: environment steps per second summed over ranks, minibatch updates per second summed
over ranks, time to target (wall seconds until the evaluation first reaches 90 percent of the single-worker
final reward, the final reward being the mean of its last ten evaluations), speedup and efficiency relative to
one worker, and the final reward. Results go to ``results/scaling.json``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from .instances import ROOT

RESULTS = ROOT / "results"
RUNS = ROOT / "runs"


def physical_cores() -> int | None:
    try:
        out = subprocess.run(["wmic", "cpu", "get", "NumberOfCores"], capture_output=True, text=True, timeout=20).stdout
        nums = [int(x) for x in out.split() if x.isdigit()]
        return sum(nums) if nums else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


_TORCHRUN_ERROR: str | None = None


def _train_args(instance: str, episodes: int, seed: int, rounds: int, out_dir: Path, tag: str, extra: list[str] | None) -> list[str]:
    return ["-m", "fleet_allocation_torch.train", "--instance", instance, "--episodes", str(episodes), "--seed", str(seed),
            "--rounds", str(rounds), "--out", str(out_dir), "--tag", tag, "--quiet", *(extra or [])]


def _run_torchrun(workers: int, train_args: list[str], env: dict, timeout: int) -> str | None:
    """Return None on success, otherwise the tail of the error output."""
    port = random.randint(29600, 29900)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={workers}", f"--master_port={port}", *train_args]
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return f"timeout after {timeout} s: {str(e)[-500:]}"
    return None if r.returncode == 0 else (r.stderr or r.stdout)[-3000:]


def _run_env_rendezvous(workers: int, train_args: list[str], env: dict, timeout: int) -> str | None:
    """Start ``workers`` ranks of the training script with the env:// rendezvous. None on success."""
    port = random.randint(29600, 29900)
    procs = []
    for rank in range(workers):
        rank_env = dict(env, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(workers), RANK=str(rank),
                        LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(workers))
        procs.append(subprocess.Popen([sys.executable, *train_args], cwd=ROOT, env=rank_env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    errors = []
    for rank, proc in enumerate(procs):
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            return f"timeout after {timeout} s in the env:// launch of {workers} ranks"
        if proc.returncode != 0:
            errors.append(f"rank {rank} exited {proc.returncode}: {(err or out)[-1500:]}")
    return "\n".join(errors) if errors else None


def torchrun_train(workers: int, instance: str, episodes: int, seed: int, rounds: int, out_dir: Path, tag: str,
                   extra: list[str] | None = None, timeout: int = 3600) -> tuple[dict | None, str | None]:
    """Train with ``workers`` ranks and return the final summary line, or the error.

    torchrun is tried first. If it fails because its rendezvous store needs libuv, the ranks are started with the
    env:// rendezvous instead and the summary records ``launcher`` and ``torchrun_error``.
    """
    global _TORCHRUN_ERROR
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.setdefault("USE_LIBUV", "0")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_args = _train_args(instance, episodes, seed, rounds, out_dir, tag, extra)
    launcher = "torchrun"
    err = _TORCHRUN_ERROR if _TORCHRUN_ERROR else _run_torchrun(workers, train_args, env, timeout)
    if err is not None and "libuv" in err:
        _TORCHRUN_ERROR = err
        launcher = "env_rendezvous"
        err = _run_env_rendezvous(workers, train_args, env, timeout)
    log = out_dir / f"{tag}.jsonl"
    if err is not None or not log.exists():
        return None, err or f"no log written at {log}"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    final = next((x for x in rows if x.get("final")), None)
    if final is None:
        return None, "no final summary line in " + str(log)
    final["evals"] = [x for x in rows if not x.get("final")]
    final["launcher"] = launcher
    final["torchrun_error"] = None if launcher == "torchrun" else _TORCHRUN_ERROR.strip().splitlines()[-1][:300]
    return final, None


def final_reward(evals: list[dict], last: int = 10) -> float:
    tail = evals[-last:] if evals else []
    return round(sum(e["eval_trips"] for e in tail) / len(tail), 2) if tail else 0.0


def time_to_target(evals: list[dict], target: float) -> float | None:
    for e in evals:
        if e["eval_trips"] >= target:
            return e["wall_s"]
    return None


def scaling_table(instance: str = "GHC_ai", episodes: int = 300, seed: int = 0, rounds: int = 10,
                  worker_counts: tuple[int, ...] = (1, 2, 4), out_file: Path | None = None, save_policy: bool = True) -> dict:
    out_dir = RUNS / "scaling"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[int, dict] = {}
    errors: dict[int, str] = {}
    for w in worker_counts:
        tag = f"{instance}_w{w}_s{seed}"
        extra = ["--save-policy"] if save_policy else []
        summary, err = torchrun_train(w, instance, episodes, seed, rounds, out_dir, tag, extra=extra)
        if summary is None:
            errors[w] = err or "unknown error"
            print(f"workers {w}: failed, {errors[w][-300:]}", flush=True)
            continue
        summaries[w] = summary
        print(f"workers {w}: {summary['env_steps_per_s']} steps/s, {summary['updates_per_s']} updates/s, "
              f"final trips {final_reward(summary['evals'])}, wall {summary['wall_s']} s", flush=True)
    rows = []
    base = summaries.get(1)
    base_final = final_reward(base["evals"]) if base else 0.0
    target = round(0.9 * base_final, 2)
    for w in worker_counts:
        s = summaries.get(w)
        if s is None:
            continue
        fr = final_reward(s["evals"])
        rows.append({
            "workers": w,
            "env_steps_per_s": s["env_steps_per_s"],
            "updates_per_s": s["updates_per_s"],
            "time_to_target_s": time_to_target(s["evals"], target),
            "speedup": round(s["env_steps_per_s"] / base["env_steps_per_s"], 3) if base else None,
            "efficiency": round(s["env_steps_per_s"] / base["env_steps_per_s"] / w, 3) if base else None,
            "final_reward": fr,
            "final_reward_vs_1_worker_pct": round(100.0 * (fr - base_final) / base_final, 2) if base_final else None,
            "wall_s": s["wall_s"],
            "env_steps_total": s["env_steps_total"],
            "updates_total": s["updates_total"],
            "naive_trips": s["naive_trips"],
            "upper_bound": s["upper_bound"],
            "launcher": s.get("launcher", "torchrun"),
        })
    result = {
        "instance": instance,
        "seed": seed,
        "episodes_per_rank": episodes,
        "rounds_per_episode": rounds,
        "threads_per_worker": 1,
        "backend": "gloo",
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "target_fraction": 0.9,
        "target_trips": target,
        "rows": rows,
        "launcher": sorted({r["launcher"] for r in rows})[0] if rows and len({r["launcher"] for r in rows}) == 1 else "mixed",
        "torchrun_error": next((s.get("torchrun_error") for s in summaries.values() if s.get("torchrun_error")), None),
        "torchrun_errors": {str(k): v for k, v in errors.items()},
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    out_file = out_file or RESULTS / "scaling.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    return result
