### 2026-03-22: TJ01 one command for v1 release regeneration

#### Executive summary

Added `lyzortx/pipeline/track_j/run_track_j.py` as a single release entry point that sequences the ST0.1 through ST0.3
foundation steps, then Track C, Track D, Track E, Track G, and Track H in dependency order. The command now
regenerates the v1 feature blocks, pair table, model, calibration artifacts, recommendations, and explained
recommendation report from raw inputs without requiring callers to stitch together individual scripts. A focused
dispatch test covers the full release path and a bounded `feature-blocks` slice.

#### Findings

- The release-critical artifacts already existed as separate track runners, so the missing piece was orchestration
  rather than new modeling logic.
- Track C needed to be sequenced explicitly because the v1 pair-table builder consumes the defense, OMP, and extended
  host feature blocks rather than materializing them itself.
- The ST0.2 pair table and ST0.3 split assignments are hard prerequisites for Track C and Track G, so the release
  command has to build that foundation before touching the v1 feature blocks.
- Track G already bootstraps any absent downstream prerequisites, so the new command can stay thin once the foundation
  is in place.

#### Interpretation

The release boundary is now executable instead of implied by the plan. One command is enough to rebuild the canonical
v1 path end to end, and the bounded `feature-blocks` mode gives a quick way to debug upstream host-feature generation
without rerunning model training.
