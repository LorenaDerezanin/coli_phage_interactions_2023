# Lyzor Tx Mission

- Build the best possible phage-lysis prediction pipeline for _E. coli_ using only in-silico methods.
- Primary local data source: `data/interactions/raw/raw_interactions.csv` and related repo metadata/features.
- No wet-lab access is assumed for this project.
- External data and literature can be added when they improve model quality or rigor.

# Operating Rules

- Keep methods reproducible and auditable (deterministic where possible).
- Use two KPI tiers:
  - **Tier 1 (Current Panel, Feasible):** evaluation with current 96-phage panel and current interaction matrix.
  - **Tier 2 (North-Star):** aspirational targets that may require panel expansion and external data.

# Steel Thread v0 Go / No-Go Gates

These are validation checkpoints, not dispatchable tasks. All must hold before expanding beyond v0:

- `run_steel_thread_v0.py` must complete without error on a fresh clone with only `phage_env` dependencies.
- No leakage violations detected by the v0 regression checks (ST0.1 through ST0.7).
- v0 model must materially outperform a naive baseline on the same split.
