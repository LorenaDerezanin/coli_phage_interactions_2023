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

### 2026-03-19: TI04 Tier A supervised ingestion priority

#### What was implemented

- Generalized `ST0.8` from a VHRdb-only ingest step to an ordered Tier A ingest step:
  `lyzortx/pipeline/steel_thread_v0/steps/st08_tier_a_ingest_ablation.py`.
- Added source-registry validation so the runnable step now enforces the planned Tier A source order
  `VHRdb -> BASEL -> KlebPhaCol -> GPB` and only ingests sources that are both present in the registry and available on
  disk.
- Extended the exported ingest schema with `source_strength_label` so GPB-style potency or strength annotations survive
  the Tier A ingest pass when present, while preserving the existing VHRdb source-fidelity fields.
- Changed the ablation summary from a two-arm VHRdb-only table to a sequential cumulative summary with
  `new_pairs_vs_internal` and `new_pairs_vs_previous_arm`.
- Added regression tests covering registry-driven source ordering, generic Tier A normalization, sequential ablation
  counts, and end-to-end output emission.

#### Findings

- Before this change, the repository had a concrete ingest path only for VHRdb even though the plan already committed to
  a broader Tier A supervised sequence.
- That mismatch would have encouraged ad hoc one-source additions later, which is exactly the wrong place to be loose:
  source ordering needs to be executable before TI05 harmonization and TI09 ablations build on it.
- The minimal generalization was to keep VHRdb-specific fidelity logic intact and define one shared contract for the
  remaining Tier A sources instead of cloning four nearly identical ingest scripts.

#### Interpretation

- Track I now has a runnable Tier A ingest spine rather than a documentation-only priority list.
- This is the right level of implementation for TI04: it encodes source order, cumulative ablation accounting, and
  provenance hooks without pretending harmonization policy is already solved before TI05.

### 2026-03-22: TI06 Tier B weak-label ingestion

#### Executive summary

TI06 now has a reproducible weak-label ingestion step under `lyzortx/pipeline/track_i/steps/` that normalizes
Virus-Host DB TSV rows and NCBI Virus/BioSample metadata into a shared provenance-preserving table. The importer keeps
raw source identifiers, expands multi-accession Virus-Host DB rows deterministically, and optionally resolves host and
phage names through the Track A canonical maps when those files are present. The output is a combined weak-label table
plus a manifest and source summary under `lyzortx/generated_outputs/track_i/tier_b_weak_label_ingest/`.

#### Findings

- Live Virus-Host DB access still exposes the expected TSV header with explicit virus and host tax IDs, names, lineage,
  PMID, evidence, sample type, and source-organism fields, which makes the source straightforward to normalize into
  one positive weak-label row per accession.
- BioSample metadata is nested rather than flat; the host signal lives in XML attributes such as `host`,
  `isolation_host`, and `isolation_source`, so the importer has to merge those attributes back onto the virus report
  row instead of assuming a single tabular export.
- BioSample `host_disease` metadata was available in the source XML but not previously surfaced in the exported
  weak-label table; the ingest step now retains it as an additional provenance field for downstream QC and slicing.
- Track A raw-name lists use pipe-delimited values, so the canonical-resolution helper needs to split `|` as well as
  commas and semicolons to make alias cross-referencing work reliably.

#### Interpretation

TI06 should be treated as source-shape normalization, not final confidence-tier assignment. The code now preserves the
weak-label provenance required for downstream TI07 tiering while keeping the pipeline honest about disagreement and
missing BioSample context instead of silently collapsing those cases.

### 2026-03-22: TI07 External-label confidence tiers

#### What was implemented

- Added `lyzortx/pipeline/track_i/steps/build_external_label_confidence_tiers.py`, a dedicated TI07 policy step that
  reads external ingest outputs and assigns a unified `external_label_confidence_tier`, numeric confidence score, and
  training weight per record.
- Kept the policy deliberately small and executable instead of inventing many bespoke bins: base confidence comes from
  source family (`A` assay-backed external sources > curated metadata knowledgebases > repository/submitter metadata),
  and row-level degradations apply for disagreement, non-`ok` QC flags, and unresolved entity mapping.
- Updated `lyzortx/pipeline/track_i/run_track_i.py` so Track I can run the new confidence-tier step directly or as part
  of `--step all`.
- Added unit tests covering the intended source-family ordering, downgrade behavior for unresolved mappings, and
  end-to-end artifact emission for mixed Tier A and Tier B inputs.

#### Findings

- A four-bin policy (`high`, `medium`, `low`, `exclude`) is enough for the current external landscape; adding finer
  subclasses now would create unstable distinctions before TI08/TI09 provide empirical feedback.
- The most important row-level degradations are provenance-quality failures rather than subtle biological heuristics:
  disagreement, bad QC states, and unresolved canonical mapping are strong enough to demote or exclude a record without
  pretending we can rescue it by hand-crafted weighting.
- This design gives TI08 a concrete interface now: every external row can be filtered or down-weighted mechanically
  without coupling integration logic to source-specific if/else blocks.

#### Interpretation

- TI07 is now an explicit policy boundary between ingestion and training. That is the right abstraction: TI06 preserves
  raw provenance, TI07 translates provenance into training trust, and TI08 can stay focused on optional integration
  rather than re-litigating confidence semantics inside the model pipeline.
