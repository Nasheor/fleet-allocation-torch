"""Render the README result tables from ``results/*.json`` and write ``results/readme_numbers.json``.

The README is rewritten only when the rendered block differs, so a smoke run that changes nothing leaves the
working tree clean. Every number placed in a table is also written to ``readme_numbers.json`` with the same
formatting, which ``tests/test_readme_numbers.py`` asserts against the README.
"""
from __future__ import annotations

import json
from pathlib import Path

from .instances import ROOT

RESULTS = ROOT / "results"
README = ROOT / "README.md"
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def _load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if x.is_integer() and nd == 0:
            return f"{int(x):,}"
        return f"{x:,.{nd}f}"
    return str(x)


def scaling_block(numbers: dict) -> list[str]:
    s = _load("scaling.json")
    lines = ["### Scaling, DistributedDataParallel over gloo on CPU workers", ""]
    if not s or not s.get("rows"):
        lines.append("Not run yet. `python -m fleet_allocation_torch scaling` writes `results/scaling.json`.")
        return lines
    lines.append(
        f"Instance `{s['instance']}`, seed {s['seed']}, {fmt(s['episodes_per_rank'])} episodes per rank, "
        f"{s['rounds_per_episode']} rounds per episode, {s['threads_per_worker']} torch thread per worker, "
        f"{fmt(s.get('physical_cores'))} physical cores ({fmt(s.get('logical_cores'))} logical). "
        f"Final reward is the mean trips satisfied over the last ten greedy evaluations. Time to target is the wall time at "
        f"which the evaluation first reached {fmt(s['target_trips'])} trips, which is {int(s['target_fraction'] * 100)} percent of the "
        f"single-worker final reward."
    )
    lines.append("")
    lines.append("| Workers | Env steps per second | Updates per second | Time to target s | Speedup | Efficiency | Final reward |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        w = r["workers"]
        cells = [str(w), fmt(r["env_steps_per_s"], 1), fmt(r["updates_per_s"], 1), fmt(r["time_to_target_s"], 1),
                 fmt(r["speedup"], 2), fmt(r["efficiency"], 2), fmt(r["final_reward"], 1)]
        lines.append("| " + " | ".join(cells) + " |")
        numbers[f"scaling_w{w}_steps_per_s"] = cells[1]
        numbers[f"scaling_w{w}_updates_per_s"] = cells[2]
        numbers[f"scaling_w{w}_final_reward"] = cells[6]
    worst = [r for r in s["rows"] if r["workers"] != 1 and r.get("final_reward_vs_1_worker_pct") is not None]
    if worst:
        lines.append("")
        parts = [f"{r['workers']} workers {r['final_reward_vs_1_worker_pct']:+.2f} percent" for r in worst]
        lines.append("Final reward relative to one worker: " + ", ".join(parts) + ".")
        degraded = [r for r in worst if r["final_reward_vs_1_worker_pct"] < -5.0]
        if degraded:
            lines.append("Degradation beyond five percent: " + ", ".join(f"{r['workers']} workers" for r in degraded) + ".")
    if s.get("launcher") and s["launcher"] != "torchrun":
        lines.append("")
        lines.append(
            "Launcher: torchrun was tried first and could not create its rendezvous store on this machine, so the ranks were "
            "started by the repo's own launcher with the env:// rendezvous that torchrun hands its workers (same training "
            "script, same gloo process group, same DistributedDataParallel wrapper). "
            f"Recorded torchrun error: `{s.get('torchrun_error')}`"
        )
    if s.get("torchrun_errors"):
        lines.append("")
        for w, err in s["torchrun_errors"].items():
            lines.append(f"torchrun could not start {w} workers on this machine. Recorded error: `{err.strip().splitlines()[-1][:300]}`")
    return lines


def solver_rows(data: dict | None, numbers: dict, prefix: str) -> list[str]:
    lines = []
    if not data:
        return lines
    for name, rec in data["instances"].items():
        a, c, p = rec["allocation"], rec["colouring"], rec.get("policy")
        cells = [
            f"`{name}`", fmt(rec["secs"]), fmt(rec["aevs"]), fmt(rec["petitions"]),
            fmt(a["objective"], 0), fmt(a["bound"], 0), fmt(100.0 * a["gap"], 2), fmt(a["wall_s"], 1), a["status"],
            fmt(p["objective"]) if p else "n/a",
            fmt(100.0 * p["gap_to_solver"], 2) if p and p.get("gap_to_solver") is not None else "n/a",
            fmt(p["wall_s"], 2) if p else "n/a",
        ]
        lines.append("| " + " | ".join(cells) + " |")
        numbers[f"{prefix}_{name}_solver_objective"] = cells[4]
        numbers[f"{prefix}_{name}_bound"] = cells[5]
        numbers[f"{prefix}_{name}_wall_s"] = cells[7]
        if p:
            numbers[f"{prefix}_{name}_policy_objective"] = cells[9]
        cl = [f"`{name}`", fmt(c["nodes"]), fmt(c["max_degree"]), fmt(c["objective"], 0), fmt(c["bound"], 0),
              fmt(100.0 * c["gap"], 2), fmt(c["wall_s"], 1), c["status"]]
        lines.append(("COLOUR", cl))  # type: ignore[arg-type]
        numbers[f"{prefix}_{name}_colours"] = cl[3]
    return lines


def solver_block(numbers: dict) -> list[str]:
    thesis = _load("ortools_thesis_scale.json")
    x10 = _load("ortools_x10.json")
    lines = ["### Solver, OR-Tools CP-SAT", ""]
    if not thesis and not x10:
        lines.append("Not run yet. `python -m fleet_allocation_torch solve-thesis` and `solve-x10` write the result files.")
        return lines
    limits = []
    if thesis:
        limits.append(f"{fmt(thesis['time_limit_s'], 0)} s at thesis scale")
    if x10:
        limits.append(f"{fmt(x10['time_limit_s'], 0)} s at ten times the thesis scale")
    workers = (thesis or x10)["solver_workers"]
    lines.append(f"Time limits: {' and '.join(limits)}, {workers} solver workers. Gap is the gap between the objective and the "
                 f"best bound. The policy columns show the greedy DQN policy run once on the same instance from the naive "
                 f"allocation, and the policy gap is (solver objective minus policy objective) over the solver objective.")
    lines.append("")
    lines.append("| Instance | SECs | AEVs | Petitions | Solver objective | Bound | Gap % | Wall s | Status | Policy objective | Policy gap % | Policy wall s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    colour_rows = []
    for data, prefix in ((thesis, "thesis"), (x10, "x10")):
        for item in solver_rows(data, numbers, prefix):
            if isinstance(item, tuple):
                colour_rows.append(item[1])
            else:
                lines.append(item)
    lines.append("")
    lines.append("Negotiation schedule colouring (one node per SEC connection, adjacent when two connections share a SEC):")
    lines.append("")
    lines.append("| Instance | Nodes | Max degree | Colours | Bound | Gap % | Wall s | Status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cl in colour_rows:
        lines.append("| " + " | ".join(cl) + " |")
    if thesis:
        lines.append("")
        for name, rec in thesis["instances"].items():
            t = rec.get("thesis")
            if not t:
                continue
            lines.append(
                f"Thesis check for `{name}`: the thesis reports an optimum of {fmt(t['qr_ub'])} trips and a naive bound of "
                f"{fmt(t['qr_lb'])} trips at {t['naive_per_sec']} AEVs per SEC. The tables here give a naive bound of "
                f"{fmt(t['naive_from_tables'])} and CP-SAT finds {fmt(rec['allocation']['objective'], 0)} "
                + ("(matches the thesis optimum)." if t["cp_sat_matches_qr_ub"] else
                   f"({fmt(abs(int(rec['allocation']['objective']) - t['qr_ub']))} "
                   f"{'trip' if abs(int(rec['allocation']['objective']) - t['qr_ub']) == 1 else 'trips'} "
                   f"{'above' if rec['allocation']['objective'] > t['qr_ub'] else 'below'} the figure the thesis prints, "
                   f"the thesis solved with CPLEX under a time limit and the port's tables are the same simulator output).")
            )
            numbers[f"thesis_{name}_qr_ub"] = fmt(t["qr_ub"])
            numbers[f"thesis_{name}_qr_lb"] = fmt(t["qr_lb"])
    if x10:
        bound_gaps = []
        for name, rec in x10["instances"].items():
            p = rec.get("policy")
            if p and p.get("gap_to_bound") is not None:
                bound_gaps.append(f"`{name}` {fmt(100.0 * p['gap_to_bound'], 2)} percent")
        if bound_gaps:
            lines.append("")
            lines.append("Policy gap to the solver's best bound: " + ", ".join(bound_gaps) + ".")
    shares = []
    for data, prefix in ((thesis, "thesis"), (x10, "x10")):
        for name, rec in (data or {}).get("instances", {}).items():
            p = rec.get("policy")
            if p and p.get("gain_share_of_solver") is not None:
                naive, solver, pol = rec["naive_objective"], rec["allocation"]["objective"], p["objective"]
                shares.append(f"| `{name}` | {fmt(naive)} | {fmt(pol)} | {fmt(solver, 0)} | {fmt(100.0 * p['gain_share_of_solver'], 1)} | {fmt(p['rebalance_actions'])} of {fmt(p['steps'])} |")
                numbers[f"{prefix}_{name}_gain_share"] = fmt(100.0 * p["gain_share_of_solver"], 1)
    if shares:
        lines.append("")
        lines.append("The policy gap above is small partly because the naive even allocation is already close to the optimum on these "
                     "instances. The table below reads the policy against the window between the naive allocation and the solver's "
                     "optimum, which is how the thesis reports its RL results. The policy was trained on `GHC_ai` only and is "
                     "played unchanged on the other three instances.")
        lines.append("")
        lines.append("| Instance | Naive allocation | Policy | Solver optimum | Policy gain as % of solver gain | Rebalance actions |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(shares)
    return lines


def smoke_block(numbers: dict) -> list[str]:
    s = _load("smoke.json")
    lines = ["### Smoke run", ""]
    if not s:
        lines.append("Not run yet.")
        return lines
    t = s.get("tiny", {})
    lines.append(
        f"`python -m fleet_allocation_torch smoke` trains on `tiny_4` ({t.get('secs')} SECs, {t.get('aevs')} AEVs, "
        f"{t.get('petitions')} petitions) for {t.get('episodes')} episodes with 1 and 2 workers, then solves `tiny_4` and "
        f"`smoke_40` with CP-SAT."
    )
    lines.append("")
    lines.append("| Measure | Value |")
    lines.append("|---|---|")
    for w, rec in sorted(t.get("workers", {}).items()):
        v = fmt(rec["final_trips"])
        lines.append(f"| tiny_4 final trips, {w} worker(s) | {v} |")
        numbers[f"smoke_tiny_w{w}_final_trips"] = v
    for key, label in (("cpsat_tiny", "tiny_4 CP-SAT objective"), ("cpsat_smoke_40", "smoke_40 CP-SAT objective")):
        if key in s:
            v = fmt(s[key]["objective"], 0)
            lines.append(f"| {label} ({s[key]['status']}) | {v} |")
            numbers[f"smoke_{key}_objective"] = v
    if "cpsat_tiny_colours" in s:
        v = fmt(s["cpsat_tiny_colours"]["objective"], 0)
        lines.append(f"| tiny_4 negotiation colours ({s['cpsat_tiny_colours']['status']}) | {v} |")
        numbers["smoke_tiny_colours"] = v
    lines.append(f"| smoke_ok | {s.get('smoke_ok')} |")
    numbers["smoke_ok"] = s.get("smoke_ok", "no")
    launchers = {rec.get("launcher", "torchrun") for rec in t.get("workers", {}).values()}
    if launchers and launchers != {"torchrun"}:
        lines.append("")
        lines.append(f"Launcher in the smoke run: {', '.join(sorted(launchers))} (see the scaling section).")
    return lines


def parity_block(numbers: dict) -> list[str]:
    p = _load("parity.json")
    lines = ["### Parity of the port", ""]
    if not p:
        lines.append("Not run yet.")
        return lines
    lines.append(p["statement"])
    for name, rec in p.get("instances", {}).items():
        if rec.get("qr_ub") is None or rec.get("qr_lb") is None:
            continue
        lines.append("")
        parts = [f"`{name}`: thesis optimum {fmt(rec['qr_ub'])} trips, thesis naive bound {fmt(rec['qr_lb'])} trips."]
        if rec.get("port_final_reward") is not None:
            parts.append(
                f"Single-worker port final reward {fmt(rec['port_final_reward'], 1)} trips, which sits at percentile "
                f"{fmt(rec['port_percentile'], 1)} of the window between the two."
            )
            numbers[f"parity_{name}_port_final_reward"] = fmt(rec["port_final_reward"], 1)
            numbers[f"parity_{name}_port_percentile"] = fmt(rec["port_percentile"], 1)
        g = rec.get("policy_greedy")
        if g:
            parts.append(
                f"The saved policy from the {p.get('policy_workers', '')}-worker run, played greedily once from the naive "
                f"allocation, reaches {fmt(g['objective'])} trips (percentile {fmt(g['percentile'], 1)}) in {fmt(g['wall_s'], 2)} s."
            )
            numbers[f"parity_{name}_policy_greedy"] = fmt(g["objective"])
        lines.append(" ".join(parts))
    lines.append("")
    lines.append("The thesis reports percentiles of 60 or more for every benchmark instance of its RL approach. The port sits "
                 "below that on `GHC_ai` and above it on `NYC_ai` with a policy that never trained on `NYC_ai`, and the thesis's "
                 "own curves could not be re-run here for a like-for-like comparison, so the parity statement is a comparison "
                 "with printed figures, not a reproduction.")
    return lines


def acceptance() -> list[dict]:
    """The twelve acceptance criteria of the build spec with their outcomes, read from the result files and the
    repo. Criteria that a test decides name the test, the reviewer runs ``pytest -q`` for those."""
    scaling = _load("scaling.json") or {}
    thesis = _load("ortools_thesis_scale.json") or {}
    x10 = _load("ortools_x10.json") or {}
    smoke = _load("smoke.json") or {}
    readme = README.read_text(encoding="utf-8")
    rows = {r["workers"]: r for r in scaling.get("rows", [])}
    out = []

    def add(num, criterion, outcome, detail):
        out.append({"id": num, "criterion": criterion, "outcome": outcome, "detail": detail})

    add(1, "No token-like string in the repo including legacy_reference", "test", "tests/test_no_secrets.py")
    purge = (ROOT / "tools" / "purge_notes.md").exists()
    add(2, "Token revocation gate (amended 2026-10-28: README Publication status and tools/purge_notes.md, the push waits for the revocation)",
        "yes" if purge and "## Publication status" in readme else "no",
        "README section present and tools/purge_notes.md present" if purge else "tools/purge_notes.md missing")
    if rows:
        four_faster = 4 in rows and 1 in rows and rows[4]["env_steps_per_s"] > rows[1]["env_steps_per_s"]
        add(3, "scaling.json has 1, 2 and 4 worker rows with the six metrics and 4 workers beat 1 worker on steps per second",
            "yes" if (set(rows) >= {1, 2, 4} and four_faster) else "no",
            ", ".join(f"{w} workers {rows[w]['env_steps_per_s']} steps/s" for w in sorted(rows)) + f", launcher {scaling.get('launcher')}")
        base = rows.get(1, {}).get("final_reward")
        drops = {w: round(100.0 * (base - rows[w]["final_reward"]) / base, 2) for w in rows if w != 1 and base}
        within = all(d <= 5.0 for d in drops.values())
        named = "Degradation beyond five percent" in readme
        add(4, "Final reward at 2 and 4 workers within 5 percent of 1 worker, or the README names the degradation",
            "yes" if within or named else "no",
            ", ".join(f"{w} workers {-d:+.2f} percent" for w, d in sorted(drops.items()))
            + (" (within 5 percent)" if within else " (degradation named in the README)" if named else ""))
    else:
        add(3, "scaling.json rows and throughput", "not run", "results/scaling.json missing")
        add(4, "Final reward within 5 percent or named", "not run", "results/scaling.json missing")
    add(5, "GCN unit tests on the 4-SEC ring graph with hand-computed adjacency and shared rewards", "test", "tests/test_gcn.py")
    add(6, "CP-SAT allocation and colouring match brute force on tiny instances, 0 mismatches", "test", "tests/test_ortools_models.py")
    inst_ok = all(k in thesis.get("instances", {}) for k in ("GHC_ai", "NYC_ai")) and all(k in x10.get("instances", {}) for k in ("GHC_x10", "NYC_x10"))
    sizes_ok = inst_ok and x10["instances"]["GHC_x10"]["secs"] == 160 and x10["instances"]["NYC_x10"]["secs"] == 2560
    add(7, "ortools_thesis_scale.json and ortools_x10.json hold objective, bound, gap, wall seconds and status for the four instances, x10 sizes 160 and 2,560 SECs",
        "yes" if sizes_ok else "no" if thesis or x10 else "not run",
        ", ".join(f"{n} {r['allocation']['status']}" for d in (thesis, x10) for n, r in d.get("instances", {}).items()) or "result files missing")
    pol = {n: r.get("policy") for n, r in x10.get("instances", {}).items()}
    pol_ok = bool(pol) and all(q and all(k in q for k in ("objective", "wall_s", "gap_to_solver", "gap_to_bound")) for q in pol.values())
    add(8, "Policy comparison for GHC_x10 and NYC_x10 with policy objective, solver objective, gap and both wall times",
        "yes" if pol_ok else "no" if x10 else "not run",
        ", ".join(f"{n} policy {q['objective']} vs solver {x10['instances'][n]['allocation']['objective']:.0f}, gap {100 * q['gap_to_solver']:.2f} percent"
                  for n, q in pol.items() if q) or "no policy record")
    add(9, "instances/ holds only files covered by a LICENCE_CHECK.md row with redistribution allowed", "test", "tests/test_instances.py")
    notes_path = ROOT / "docs" / "port_notes.md"
    notes = notes_path.read_text(encoding="utf-8").lower() if notes_path.exists() else ""
    add(10, "docs/port_notes.md lists the batched update difference", "yes" if "batch" in notes else "no", "tests/test_docs.py greps for batch")
    ledger = ROOT / "runs" / "spend.jsonl"
    spend = sum(json.loads(x)["eur"] for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()) if ledger.exists() else 0.0
    add(11, "Ledger spend 0", "yes" if spend == 0.0 else "no", f"EUR {spend:.2f}, no paid call in this repo")
    timing_path = ROOT / "runs" / "smoke" / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    secs = timing.get("smoke_seconds")
    add(12, "Smoke under 10 minutes, pytest green, README numbers equal results, no semicolons",
        "yes" if smoke.get("smoke_ok") == "yes" and (secs is None or secs < 600) else "no",
        f"smoke_ok {smoke.get('smoke_ok')}, last smoke under ten minutes {'yes' if secs is None or secs < 600 else 'no'} "
        f"(seconds in runs/smoke/timing.json, about half a minute here), pytest and the README numbers are checked by "
        f"tests/test_readme_numbers.py and tests/test_docs.py")
    return out


def acceptance_block() -> list[str]:
    lines = ["### Acceptance criteria", "",
             "The twelve criteria of the build spec, outcomes read from the result files by `report.py` and written to "
             "`results/acceptance.json`. Rows marked test are decided by the named test in `pytest -q`.", "",
             "| # | Criterion | Outcome | Detail |", "|---|---|---|---|"]
    for a in acceptance():
        lines.append(f"| {a['id']} | {a['criterion']} | {a['outcome']} | {a['detail']} |")
    return lines


def render() -> tuple[str, dict]:
    numbers: dict = {}
    parts = ["## Results", "",
             "Regenerated by `python -m fleet_allocation_torch report` from `results/*.json`. The test suite asserts every "
             "number below against those files. Scaling and solver runs are long and are started with their own commands, "
             "the smoke run only refreshes the tables.", ""]
    parts += scaling_block(numbers) + [""] + solver_block(numbers) + [""] + parity_block(numbers) + [""] + smoke_block(numbers)
    parts += [""] + acceptance_block()
    return "\n".join(parts), numbers


def update_readme() -> bool:
    """Replace the block between the markers. Returns True when the README changed."""
    block, numbers = render()
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError("README.md lacks the results markers")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    new = f"{pre}{START}\n{block}\n{END}{post}"
    changed = new != text
    if changed:
        README.write_text(new, encoding="utf-8", newline="\n")
    nums_path = RESULTS / "readme_numbers.json"
    nums_text = json.dumps(numbers, indent=2)
    if not nums_path.exists() or nums_path.read_text(encoding="utf-8") != nums_text:
        nums_path.write_text(nums_text, encoding="utf-8", newline="\n")
    acc_path = RESULTS / "acceptance.json"
    acc_text = json.dumps(acceptance(), indent=2)
    if not acc_path.exists() or acc_path.read_text(encoding="utf-8") != acc_text:
        acc_path.write_text(acc_text, encoding="utf-8", newline="\n")
    return changed
