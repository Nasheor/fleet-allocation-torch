"""Command line entry point.

``python -m fleet_allocation_torch smoke`` trains on the tiny instance with 1 and 2 workers, solves two small
instances with CP-SAT, writes ``results/smoke.json`` and ``results/readme_numbers.json`` and refreshes the README
tables, all in under ten minutes. The long runs have their own commands: ``scaling``, ``solve-thesis``,
``solve-x10`` and ``parity``. ``instances`` rebuilds the instance files and ``report`` only re-renders the tables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .instances import INSTANCES, ROOT, load_instance

RESULTS = ROOT / "results"
RUNS = ROOT / "runs"


def _write_if_changed(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8", newline="\n")


def ensure_instances() -> None:
    from .build_instances import build_all

    needed = ["tiny_4", "smoke_40"]
    if any(not (INSTANCES / f"{n}.json").exists() for n in needed):
        build_all()


def smoke() -> int:
    from .ortools_models import line_graph, solve_allocation, solve_colouring
    from .report import update_readme
    from .scaling import torchrun_train

    t0 = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    ensure_instances()
    tiny = load_instance("tiny_4")
    out: dict = {
        "tiny": {"secs": tiny.num_secs, "aevs": tiny.num_evs, "petitions": tiny.num_petitions, "episodes": 20, "rounds": 10,
                 "naive_trips": int(tiny.total_value(tiny.naive_allocation())), "workers": {}},
    }
    errors = {}
    for w in (1, 2):
        summary, err = torchrun_train(w, "tiny_4", episodes=20, seed=0, rounds=10, out_dir=RUNS / "smoke", tag=f"smoke_tiny_w{w}", timeout=600)
        if summary is None:
            errors[str(w)] = err
            print(f"smoke: {w} worker(s) failed: {err[-500:] if err else err}", flush=True)
            continue
        out["tiny"]["workers"][str(w)] = {
            "final_trips": summary["eval_trips"],
            "env_steps_total": summary["env_steps_total"],
            "updates_total": summary["updates_total"],
            "launcher": summary.get("launcher", "torchrun"),
        }
        if summary.get("torchrun_error"):
            out["torchrun_error"] = summary["torchrun_error"]
        print(f"smoke: {w} worker(s): {summary['env_steps_per_s']} steps/s, final trips {summary['eval_trips']}", flush=True)
    a = solve_allocation(tiny.values, tiny.num_evs, ub=tiny.upper_bound(), time_limit=30, workers=8)
    out["cpsat_tiny"] = {"objective": a.objective, "bound": a.bound, "status": a.status, "allocated_aevs": int(sum(a.solution))}
    n, adj = line_graph(tiny.num_secs, tiny.connections)
    c = solve_colouring(n, adj, time_limit=30, workers=8)
    out["cpsat_tiny_colours"] = {"objective": c.objective, "bound": c.bound, "status": c.status}
    big = load_instance("smoke_40")
    b = solve_allocation(big.values, big.num_evs, ub=big.upper_bound(), time_limit=30, workers=8)
    out["cpsat_smoke_40"] = {"objective": b.objective, "bound": b.bound, "gap": round(b.gap, 6), "status": b.status,
                             "secs": big.num_secs, "aevs": big.num_evs, "naive_trips": int(big.total_value(big.naive_allocation()))}
    print(f"smoke: CP-SAT tiny {a.objective} ({a.status}), colours {c.objective} ({c.status}), smoke_40 {b.objective} ({b.status}) in {b.wall_s:.1f} s", flush=True)
    ok = not errors and a.status == "OPTIMAL" and b.status in ("OPTIMAL", "FEASIBLE") and "1" in out["tiny"]["workers"]
    out["smoke_ok"] = "yes" if ok else "no"
    if errors:
        out["torchrun_errors"] = errors
    _write_if_changed(RESULTS / "smoke.json", out)
    changed = update_readme()
    timing = {"smoke_seconds": round(time.perf_counter() - t0, 1), "readme_changed": changed}
    (RUNS / "smoke").mkdir(parents=True, exist_ok=True)
    (RUNS / "smoke" / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    print(f"smoke ok={out['smoke_ok']} in {timing['smoke_seconds']} s, README changed: {changed}", flush=True)
    return 0 if ok else 1


def parity(policy_path: Path | None = None) -> int:
    """Parity record of the port. The legacy TensorFlow agent is not importable on this machine, so the port is
    compared with the figures the thesis prints for GHC_ai and NYC_ai."""
    from .report import update_readme

    scaling = json.loads((RESULTS / "scaling.json").read_text(encoding="utf-8")) if (RESULTS / "scaling.json").exists() else None
    try:
        import tensorflow  # noqa: F401

        tf_ok = True
    except ImportError:
        tf_ok = False
    out = {
        "legacy_reproduced": False,
        "tensorflow_importable": tf_ok,
        "statement": ("Legacy curve not reproduced on this machine: TensorFlow is not importable in any environment here, so the "
                      "original Keras agent was not run. The port is compared with the reward figures the thesis prints instead."),
        "instances": {},
    }
    for name in ("GHC_ai", "NYC_ai"):
        inst = load_instance(name)
        rec = {"qr_ub": inst.thesis.get("qr_ub"), "qr_lb": inst.thesis.get("qr_lb"),
               "naive_from_tables": int(inst.total_value(inst.naive_allocation())), "sum_of_maxima": inst.upper_bound()}
        if scaling and scaling["instance"] == name and scaling["rows"]:
            row = scaling["rows"][0]
            fr = row["final_reward"]
            rec["port_final_reward"] = fr
            rec["port_workers"] = row["workers"]
            if rec["qr_ub"] and rec["qr_lb"]:
                rec["port_percentile"] = round(100.0 * (fr - rec["qr_lb"]) / (rec["qr_ub"] - rec["qr_lb"]), 2)
        out["instances"][name] = rec
    if policy_path and policy_path.exists():
        from .solve import greedy_episode, load_policy

        net, meta = load_policy(policy_path)
        out["policy_source"] = policy_path.name
        out["policy_workers"] = meta.get("world_size")
        for name in ("GHC_ai", "NYC_ai"):
            g = greedy_episode(net, load_instance(name))
            rec = out["instances"][name]
            if rec.get("qr_ub") and rec.get("qr_lb"):
                g["percentile"] = round(100.0 * (g["objective"] - rec["qr_lb"]) / (rec["qr_ub"] - rec["qr_lb"]), 2)
            rec["policy_greedy"] = g
    out["instances"] = {k: v for k, v in out["instances"].items() if "port_final_reward" in v or "policy_greedy" in v}
    _write_if_changed(RESULTS / "parity.json", out)
    update_readme()
    print(json.dumps(out, indent=2))
    return 0


def _policy_path(arg: str | None) -> Path | None:
    if arg:
        return Path(arg)
    candidates = sorted(RESULTS.glob("policy_GHC_ai_w*_s0.json"), key=lambda p: p.name, reverse=True)
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fleet_allocation_torch")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="short reproducible run that regenerates results/ and the README tables")
    sub.add_parser("instances", help="rebuild the instance files under instances/")
    sc = sub.add_parser("scaling", help="1, 2 and 4 worker runs on GHC_ai and the scaling table")
    sc.add_argument("--episodes", type=int, default=300)
    sc.add_argument("--instance", default="GHC_ai")
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--rounds", type=int, default=10)
    st = sub.add_parser("solve-thesis", help="CP-SAT on GHC_ai and NYC_ai")
    st.add_argument("--time-limit", type=float, default=600.0)
    st.add_argument("--policy", default=None)
    sx = sub.add_parser("solve-x10", help="CP-SAT on GHC_x10 and NYC_x10 plus the policy comparison")
    sx.add_argument("--time-limit", type=float, default=1800.0)
    sx.add_argument("--policy", default=None)
    pa = sub.add_parser("parity", help="parity record of the port against the thesis figures")
    pa.add_argument("--policy", default=None)
    sub.add_parser("report", help="re-render the README tables from results/")
    args = p.parse_args(argv)

    if args.cmd == "smoke":
        return smoke()
    if args.cmd == "instances":
        from .build_instances import build_all

        for path in build_all():
            print(path)
        return 0
    if args.cmd == "scaling":
        from .report import update_readme
        from .scaling import scaling_table
        from .solve import save_policy_json

        res = scaling_table(instance=args.instance, episodes=args.episodes, seed=args.seed, rounds=args.rounds)
        best = max((r["workers"] for r in res["rows"]), default=None)
        if best is not None:
            pt = RUNS / "scaling" / f"{args.instance}_w{best}_s{args.seed}.pt"
            if pt.exists():
                print("policy saved to", save_policy_json(pt, RESULTS / f"policy_{args.instance}_w{best}_s{args.seed}.json"))
        update_readme()
        return 0
    if args.cmd == "solve-thesis":
        from .report import update_readme
        from .solve import thesis_scale

        thesis_scale(time_limit=args.time_limit, policy_path=_policy_path(args.policy))
        update_readme()
        return 0
    if args.cmd == "solve-x10":
        from .report import update_readme
        from .solve import x10_scale

        x10_scale(time_limit=args.time_limit, policy_path=_policy_path(args.policy))
        update_readme()
        return 0
    if args.cmd == "parity":
        return parity(_policy_path(args.policy))
    if args.cmd == "report":
        from .report import update_readme

        print("README changed:", update_readme())
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
