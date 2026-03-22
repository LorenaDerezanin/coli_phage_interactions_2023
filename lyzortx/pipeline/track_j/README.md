# Track J

`python lyzortx/pipeline/track_j/run_track_j.py`

This command regenerates the v1 release sequence from raw data in dependency order:

1. ST0.1 through ST0.3 foundation artifacts
2. Track C feature blocks and v1 pair table
3. Track D phage feature blocks
4. Track E pairwise compatibility features
5. Track G model, calibration, ablations, SHAP, and subset sweep
6. Track H explained recommendations report

Use `--step` to rerun a bounded slice when debugging:

1. `foundation`
2. `feature-blocks`
3. `modeling`
4. `recommendations`
