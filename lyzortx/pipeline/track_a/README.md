# Track A Pipeline

Track A builds a reproducible data integrity and label generation foundation from raw interaction inputs.

## Scope

- Canonical IDs for bacteria/phages across raw interactions, matrix, host metadata, phage metadata, and CV groups.
- Alias normalization outputs and candidate mismatch reports.
- Automated row/column integrity checks and integrity status report.
- Cohort contracts and denominator rules for `raw369`, `matrix402`, and `features404`.
- Observation-level intermediates preserving replicate and dilution structure.
- Label set v1 and v2 plus v1-v2 impact comparison.
- Plaque-image-assisted QC queue for ambiguous/conflicting pairs.

## Run

Run from repository root.

```bash
python3 -m lyzortx.pipeline.track_a.run_track_a --step all
```

Run build only:

```bash
python3 -m lyzortx.pipeline.track_a.run_track_a --step build
```

Run checks only (and fail on warning-level failures):

```bash
python3 -m lyzortx.pipeline.track_a.run_track_a --step check --fail-on-warnings
```

You can also run the check directly:

```bash
python3 -m lyzortx.pipeline.track_a.checks.check_track_a_integrity --run-build
```

## Outputs

- `lyzortx/generated_outputs/track_a/id_map/`
- `lyzortx/generated_outputs/track_a/integrity/`
- `lyzortx/generated_outputs/track_a/cohort/`
- `lyzortx/generated_outputs/track_a/labels/`
- `lyzortx/generated_outputs/track_a/qc/`
- `lyzortx/generated_outputs/track_a/track_a_manifest.json`

## Checkbox Mapping

- Canonical ID map: `id_map/bacteria_id_map.csv`, `id_map/phage_id_map.csv`
- Alias mismatch handling: `id_map/*alias_resolution.csv`, `id_map/*alias_candidates.csv`
- Automated integrity checks: `integrity/integrity_checks.csv`, `integrity/integrity_report.json`
- `score='n'` policy docs: `labels/label_set_v1_policy.json`, `labels/label_set_v2_policy.json`
- Cohort contracts: `cohort/cohort_contracts.csv`, `cohort/cohort_contracts.json`
- Replicate+dilution preservation: `labels/track_a_observations_with_ids.csv`,
  `labels/track_a_pair_dilution_summary.csv`, `labels/track_a_pair_observation_grid.csv`
- Label set v1: `labels/label_set_v1_pairs.csv`
- Label set v2 + comparison: `labels/label_set_v2_pairs.csv`, `labels/label_set_v1_v2_comparison.csv`
- Plaque-image-assisted QC queue: `qc/plaque_image_qc_queue.csv`, `qc/plaque_image_qc_summary.json`
