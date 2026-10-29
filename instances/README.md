# Instances

Every file here is released under the licence rows of `LICENCE_CHECK.md` (checked by `tests/test_instances.py`).
An instance is a JSON record with the SEC positions, the SEC connections (zero based), the fleet size, the
petition count and two tables with one row per SEC and one column per AEV count from 0 to the cap: `values`
(trip petitions satisfied) and `energy` (energy consumed). The cap is four times the mean allocation, where
every thesis curve is flat, and a look-up beyond it returns the value at the cap.

| File | SECs | AEVs | Petitions | Connections | Cap | Look-up tables from | Released under |
|---|---|---|---|---|---|---|---|
| `GHC_ai.json` | 16 | 400 | 10,000 | 16 | 100 | the thesis ride-sharing simulator run on the Google Hash Code 2018 `d_metropolis.in` alignment (January 2024 run) | Apache 2.0 notice for the source data, see below |
| `NYC_ai.json` | 256 | 1,536 | 50,000 | 256 | 24 | the thesis ride-sharing simulator run on the NYC 2018 green taxi alignment (January 2024 run) | NYC Open Data source identification, see below |
| `GHC_x10.json` | 160 | 4,000 | 100,000 | 160 | 100 | saturating curves resampled from the `GHC_ai` tables by `generator.py` (seed 0) | as `GHC_ai` |
| `NYC_x10.json` | 2,560 | 15,360 | 500,000 | 2,560 | 24 | saturating curves resampled from the `NYC_ai` tables by `generator.py` (seed 0) | as `NYC_ai` |
| `tiny_4.json` | 4 | 24 | 300 | 4 | 24 | synthetic, `generator.synthetic` (seed 0) | MIT |
| `smoke_40.json` | 40 | 1,000 | 25,000 | 40 | 100 | synthetic, `generator.synthetic` (seed 0) | MIT |

## Source lines

- **Google Hash Code 2018, qualification round, `d_metropolis.in`.** Copyright Google, released in the
  archive at https://github.com/google/coding-competitions-archive under the Apache License, Version 2.0.
  No part of the input file is reproduced here. `GHC_ai` carries the 16 SEC positions and connections of the
  thesis alignment and the per-SEC counts of rides served that the thesis simulator computed for 1 to 100
  vehicles per SEC. `GHC_x10` carries curves fitted to those counts.
- **NYC Taxi and Limousine Commission trip record data, 2018 green taxi trips, and the NYC Taxi Zones.**
  Source: NYC Open Data, published by the Taxi and Limousine Commission under the NYC Open Data Law (Local
  Law 11 of 2012), which requires data sets to be available without licence requirement or usage
  restrictions and lets the City ask a re-publisher to identify the source, version and modifications.
  Version: the 2018 trip records as downloaded for the thesis in 2023. Modifications: a sample of one day's
  trips was aligned to a 1,000 by 1,000 grid with 256 SECs by the thesis instance generator, and the thesis
  simulator computed the per-SEC counts of trips served for 1 to 24 vehicles per SEC. No trip row, time,
  location or fare is reproduced here. `NYC_x10` carries curves fitted to those counts.

## Rebuilding

```
.venv\Scripts\python -m fleet_allocation_torch instances
```

The two thesis-scale instances need the simulator outputs and instance headers of the thesis working tree under
`data/raw/thesis/Metropolis/` and `data/raw/thesis/NYC/` (`subproblem_solutions.csv` and `input.in`, not
tracked). Without them the command keeps the committed files and rebuilds only the generated ones. The ten
times scale instances take the thesis-scale files as their reference and are deterministic for a given seed.
