### 2026-03-17: TI03 VHRdb source-fidelity fields

#### What was implemented

- Updated `ST0.8` VHRdb ingest so ingested rows now carry raw `global_response` and `datasource_response` columns
  directly, instead of only preserving those values under `source_*response` aliases.
- Kept `label_hard_any_lysis` derived from the normalized global response for downstream compatibility, while preserving
  the original mixed-case source values in the exported ingest table.
- Added a file-boundary regression test for `main()` to verify that the emitted
  `st08_vhrdb_ingested_pairs.csv` preserves raw response strings and populates `source_datasource_id`,
  `source_native_record_id`, and `source_disagreement_flag`.

#### Findings

- Before this change, the ingest step already retained raw VHRdb response strings in memory, but only under
  `source_global_response` / `source_datasource_response`; the exported ingest schema did not expose the raw values
  under the original column names required by the task acceptance criteria.
- The source metadata columns were already populated for VHRdb rows, so the main implementation gap was output-schema
  fidelity rather than missing provenance logic.

#### Interpretation

- `ST0.8` now preserves both downstream usability and source auditability: the normalized label remains available for
  modeling, while the raw VHRdb response fields survive unchanged in the ingested table for provenance checks and later
  harmonization work.
