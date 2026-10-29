"""Build the released instances under ``instances/``.

Thesis scale: ``GHC_ai`` (16 SECs, 400 AEVs, 10,000 petitions) and ``NYC_ai`` (256 SECs, 1,536 AEVs,
50,000 petitions) from the simulator look-up tables and the instance headers of the thesis working tree,
expected under ``data/raw/thesis/<Metropolis|NYC>/`` (gitignored). Ten times scale: ``GHC_x10`` and
``NYC_x10`` from the generator with the thesis-scale instance as the reference. Smoke: ``tiny_4`` and
``smoke_40`` from the synthetic generator. A file is rewritten only when its content changes.
"""
from __future__ import annotations

from pathlib import Path

from .generator import generate, synthetic
from .instances import INSTANCES, ROOT, Instance, instance_from_legacy, load_instance

RAW = ROOT / "data" / "raw" / "thesis"

THESIS = {
    "GHC_ai": {"qr_ub": 5196, "qr_lb": 4679, "naive_per_sec": 25, "page": 83},
    "NYC_ai": {"qr_ub": 35724, "qr_lb": 34091, "naive_per_sec": 6, "page": 83},
}


def build_thesis_scale() -> list[Instance]:
    out = []
    specs = [
        ("GHC_ai", "Metropolis", "Google Hash Code 2018 qualification input d_metropolis.in aligned by the thesis generator, "
                                 "look-up tables from the thesis ride-sharing simulator (Jan 2024 run)"),
        ("NYC_ai", "NYC", "NYC TLC 2018 green taxi trips aligned by the thesis generator on the NYC taxi zones, "
                          "look-up tables from the thesis ride-sharing simulator (Jan 2024 run)"),
    ]
    for name, folder, source in specs:
        csv = RAW / folder / "subproblem_solutions.csv"
        header = RAW / folder / "input.in"
        if not csv.exists() or not header.exists():
            print(f"skip {name}: {csv} or {header} missing")
            continue
        inst = instance_from_legacy(name, source, csv, header, thesis=THESIS[name],
                                    notes="Look-up range capped at four times the mean allocation, where every curve is flat.")
        out.append(inst)
    return out


def build_all(seed: int = 0) -> list[Path]:
    INSTANCES.mkdir(exist_ok=True)
    written = []
    thesis = {i.name: i for i in build_thesis_scale()}
    for inst in thesis.values():
        written.append(inst.save())
    for name in ("GHC_ai", "NYC_ai"):
        ref = thesis.get(name)
        if ref is None and (INSTANCES / f"{name}.json").exists():
            ref = load_instance(name)
        if ref is None:
            print(f"skip {name[:3]}_x10: no reference instance")
            continue
        written.append(generate(f"{name[:3]}_x10", ref, factor=10, seed=seed).save())
    written.append(synthetic("tiny_4", num_secs=4, num_evs=24, num_petitions=300, seed=seed).save())
    written.append(synthetic("smoke_40", num_secs=40, num_evs=1000, num_petitions=25000, seed=seed).save())
    return written


if __name__ == "__main__":
    for p in build_all():
        inst = load_instance(p)
        print(f"{p.name}: {inst.num_secs} SECs, {inst.num_evs} AEVs, {inst.num_petitions} petitions, cap {inst.cap}, "
              f"{len(inst.connections)} connections, naive {inst.total_value(inst.naive_allocation())}, bound {inst.upper_bound()}")
