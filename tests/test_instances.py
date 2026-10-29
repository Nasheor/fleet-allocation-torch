"""Every file under instances/ is covered by a LICENCE_CHECK.md row whose redistribution answer starts with yes."""
import fnmatch
import json
import re


def covering_patterns(repo_root):
    rows = []
    for line in (repo_root / "LICENCE_CHECK.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Item"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        redistribution = cells[4].lower()
        covers = re.findall(r"`([^`]+)`", cells[6])
        if redistribution.startswith("yes"):
            rows.extend(covers)
    return rows


def test_every_instance_file_is_released_by_an_allowing_row(repo_root):
    patterns = covering_patterns(repo_root)
    assert patterns, "no allowing rows with Covers patterns in LICENCE_CHECK.md"
    uncovered = []
    for p in sorted((repo_root / "instances").glob("*.json")):
        if not any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
            uncovered.append(p.name)
    assert not uncovered, f"instance files without an allowing licence row: {uncovered}"


def test_instance_files_parse_and_match_their_sizes(repo_root):
    expected = {"GHC_ai": (16, 400, 10000), "NYC_ai": (256, 1536, 50000), "GHC_x10": (160, 4000, 100000), "NYC_x10": (2560, 15360, 500000)}
    for name, (secs, evs, tps) in expected.items():
        data = json.loads((repo_root / "instances" / f"{name}.json").read_text(encoding="utf-8"))
        assert data["num_secs"] == secs and data["num_evs"] == evs and data["num_petitions"] == tps
        assert len(data["values"]) == secs and len(data["values"][0]) == data["cap"] + 1
        assert data["cap"] == 4 * (evs // secs)
