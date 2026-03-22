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

### 2026-03-22: TJ02 freeze environment specs and seeds for v1 benchmark run

#### Executive summary

Pinned the benchmark runtime to exact versions in `requirements.txt` and `environment.yml` so the `phage_env`
bootstrap is reproducible instead of floating on minimum-version ranges. I also documented the seed values used by the
v1 benchmark path so model training, calibration, and bootstrap reporting all share a single deterministic seed
contract.

#### Findings

- The live environment snapshot for this task was `python=3.12.12` and `pip==26.0.1`.
- Direct Python dependencies are now fully pinned to the versions currently used in the benchmark runtime, including
  `numpy==2.4.2`, `pandas==3.0.1`, `scikit-learn==1.8.0`, `lightgbm==4.6.0`, `shap==0.51.0`, and the notebook and
  linting toolchain.
- The reproducibility-critical seeds are already centralized in the v1 path:
  - `42` for `st04.random_state`, `st05.platt_random_state`, `build_v1_host_feature_pair_table.py`
    `--lightgbm-random-state`, `train_v1_binary_classifier.py` `--random-state`,
    `calibrate_gbm_outputs.py` `--platt-random-state`, `calibrate_gbm_outputs.py` `--bootstrap-random-state`, and
    the TG03/TG04/TG05-style downstream bootstrap and refit helpers
  - `0` for the Track D embedding reducers in `build_phage_genome_kmer_features.py` and
    `build_phage_distance_embedding.py`

#### Interpretation

The benchmark run is now frozen at the environment and seed level rather than only at the pipeline logic level. That
should make v1 reruns materially easier to compare, because dependency drift and stochastic training variance are both
constrained to a known baseline.
