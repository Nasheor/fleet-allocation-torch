"""Acceptance checks on the committed result files."""
import json


def _load(repo_root, name):
    return json.loads((repo_root / "results" / name).read_text(encoding="utf-8"))


def test_scaling_rows_and_throughput(repo_root):
    s = _load(repo_root, "scaling.json")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    rows = {r["workers"]: r for r in s["rows"]}
    keys = {"env_steps_per_s", "updates_per_s", "time_to_target_s", "speedup", "efficiency", "final_reward"}
    for w, r in rows.items():
        assert keys <= set(r), f"missing keys for {w} workers"
    assert 1 in rows and 2 in rows
    if 4 in rows:
        assert rows[4]["env_steps_per_s"] > rows[1]["env_steps_per_s"]
    else:
        assert "4" in s.get("torchrun_errors", {}), "no 4-worker row and no recorded torchrun error"
        assert "torchrun could not start 4 workers" in readme
    assert rows[2]["env_steps_per_s"] > rows[1]["env_steps_per_s"]


def test_final_reward_within_five_percent_or_named(repo_root):
    s = _load(repo_root, "scaling.json")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    base = next(r for r in s["rows"] if r["workers"] == 1)["final_reward"]
    for r in s["rows"]:
        if r["workers"] == 1:
            continue
        drop = (base - r["final_reward"]) / base
        if drop > 0.05:
            assert f"Degradation beyond five percent" in readme and f"{r['workers']} workers" in readme


def test_ortools_result_files_and_x10_sizes(repo_root):
    for name, expected in (("ortools_thesis_scale.json", {"GHC_ai": 16, "NYC_ai": 256}), ("ortools_x10.json", {"GHC_x10": 160, "NYC_x10": 2560})):
        data = _load(repo_root, name)
        for inst, secs in expected.items():
            rec = data["instances"][inst]
            assert rec["secs"] == secs
            for model in ("allocation", "colouring"):
                for k in ("objective", "bound", "gap", "wall_s", "status"):
                    assert k in rec[model], f"{name} {inst} {model} lacks {k}"


def test_policy_comparison_at_x10(repo_root):
    data = _load(repo_root, "ortools_x10.json")
    for inst in ("GHC_x10", "NYC_x10"):
        rec = data["instances"][inst]
        p = rec["policy"]
        for k in ("objective", "wall_s", "gap_to_solver", "gap_to_bound"):
            assert k in p
        assert rec["allocation"]["wall_s"] >= 0 and p["wall_s"] >= 0
        assert p["objective"] >= rec["naive_objective"]


def test_thesis_scale_matches_the_thesis_figures(repo_root):
    data = _load(repo_root, "ortools_thesis_scale.json")
    for inst in ("GHC_ai", "NYC_ai"):
        t = data["instances"][inst]["thesis"]
        assert t["naive_matches_qr_lb"]


def test_ledger_spend_is_zero(repo_root):
    ledger = repo_root / "runs" / "spend.jsonl"
    total = 0.0
    if ledger.exists():
        total = sum(json.loads(x)["eur"] for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip())
    assert total == 0.0
