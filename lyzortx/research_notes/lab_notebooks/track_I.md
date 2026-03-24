### 2026-03-24: TI03 Download and ingest VHRdb pairs with source-fidelity fields

#### Executive summary

TI03 now has a real Track I download-and-ingest step under
`lyzortx/pipeline/track_i/steps/build_tier_a_vhrdb_ingest.py`. The step downloads the maintained public VHRdb API
artifacts, writes the raw JSON payloads plus a normalized CSV under
`lyzortx/generated_outputs/track_i/tier_a_ingest/`, and fails loudly on request or file-missing errors. A live run on
2026-03-24 produced 56,672 ingested public virus-host-datasource rows with preserved `global_response` and
`datasource_response` fields plus populated `source_datasource_id`, `source_native_record_id`, and
`source_disagreement_flag`.

#### What changed

- Added a dedicated TI03 Track I step instead of reusing the old `ST0.8` placeholder path.
- The step downloads six raw VHRdb API artifacts into
  `lyzortx/generated_outputs/track_i/tier_a_ingest/raw_vhrdb_downloads/`: global response labels, datasource
  metadata, virus metadata, host metadata, per-datasource responses, and aggregated responses.
- Normalization emits `ti03_vhrdb_ingested_pairs.csv`, `ti03_vhrdb_ingest_summary.csv`, and
  `ti03_vhrdb_ingest_manifest.json` under `lyzortx/generated_outputs/track_i/tier_a_ingest/`.
- Updated `lyzortx/pipeline/track_i/run_track_i.py` so Track I can run TI03 directly via `--step tier-a-ingest` and
  as part of `--step all`.
- Updated TI07's default Tier A input path to the TI03 artifact instead of the stale steel-thread output.

#### Source notes

- Official VHRdb API docs:
  `https://hub.pages.pasteur.fr/viralhostrangedb/api.html`
  Quote: "The format of the data is virus_id>host_id>datasource_id>value."
- Same docs:
  `https://hub.pages.pasteur.fr/viralhostrangedb/api.html`
  Quote: "presented responses are the responses within the global scheme, not the raw responses."

#### Findings

- The CityU `phage.ee.cityu.edu.hk` landing page listed generic PhaBOX downloads but did not expose a maintained
  VHRdb response-level export. The maintained public VHRdb API lives at `viralhostrangedb.pasteur.cloud`, so the
  source registry entry now points there instead of the stale CityU URL.
- The public API shape is sufficient for TI03: aggregated responses provide the pair-level global response, while the
  per-datasource response payload provides the source-specific response needed for row-level provenance.
- The live 2026-03-24 ingest produced 56,672 rows. The exported response mix was 45,424 `No infection`,
  3,376 `Intermediate`, and 7,872 `Infection`.
- `source_disagreement_flag` is now computed from the emitted public datasource rows for each virus-host pair rather
  than guessed from a precomputed count field.

#### Interpretation

- Track I now has a real Tier A entry point instead of a documentation-only placeholder. This fixes the root failure
  that invalidated the earlier TI03-TI10 chain: the pipeline now downloads public VHRdb data itself and materializes a
  concrete ingest artifact with traceable provenance.
- The preserved response fields are VHRdb's public global-scheme and per-datasource response labels, which is the
  highest-fidelity public representation exposed by the maintained API. That is sufficient for the acceptance criteria
  and for downstream confidence-tiering and ablation work.

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

### 2026-03-24: TI04 Download and ingest Tier A sources BASEL, KlebPhaCol, and GPB

#### Executive summary

TI04 now has a concrete download-and-ingest step under
`lyzortx/pipeline/track_i/steps/build_tier_a_additional_source_ingests.py` that materializes three new Tier A CSVs
under `lyzortx/generated_outputs/track_i/tier_a_ingest/`. A live run on 2026-03-24 produced 468 BASEL rows, 7,697
KlebPhaCol rows, and 3,960 GPB rows, all with populated `source_system` provenance and failure-on-missing-data
behavior. Two nominal source endpoints were broken in practice, so the implementation uses the live official artifacts
that actually expose host-range data and records that fallback in the source registry.

#### What changed

- Added a dedicated TI04 step that downloads three raw artifacts into
  `lyzortx/generated_outputs/track_i/tier_a_ingest/raw_ti04_downloads/`: the BASEL PLOS S1 workbook, the KlebPhaCol
  `site-data.json` export, and the GPB Nature source-data workbook.
- Normalization writes one pair table, summary CSV, and manifest JSON per source:
  `ti04_basel_*`, `ti04_klebphacol_*`, and `ti04_gpb_*`.
- Updated `lyzortx/pipeline/track_i/run_track_i.py` so `--step tier-a-ingest` now runs both TI03 and TI04 instead of
  only the VHRdb ingest.
- Refreshed `lyzortx/research_notes/external_data/source_registry.csv` to point at the live, machine-readable
  artifacts that were verified on 2026-03-24.

#### Findings

- The PMC mirror for BASEL exposed a "Preparing to download" HTML gate instead of the workbook, but the canonical PLOS
  supplementary download URL for S1 Data returned the real XLSX payload. The TI04 parser reads the qualitative host
  range columns for the six enterobacterial strains directly from that workbook.
- `https://klebphacol.com/` did not resolve on 2026-03-24. The live public host-range frontend at
  `https://phage.klebphacol.soton.ac.uk/` loads `site-data.json`, which contains explicit per-phage lists of hosts
  with lysis, no lysis, and undetermined lysis in LB and TSB media.
- `https://phagebank.org/` returned a service-provider error on 2026-03-24. The corresponding Nature paper source-data
  workbook still exposed the GPB host-range matrix on sheet `figure3abfh`, with response codes `0/1/2/3` representing
  no infection, anoxic-only infection, oxic-only infection, and infection in both conditions.
- The live run produced the following row counts and unique pair counts:
  BASEL `468/468`, KlebPhaCol `7,697/3,848`, GPB `3,960/3,960` (rows/pairs). KlebPhaCol has more rows than unique
  pairs because the same phage-host pair can appear in multiple media and with disagreement across media.

#### Interpretation

TI04 now satisfies the previously missing substance requirement for these Tier A sources: Track I downloads real public
artifacts and emits non-empty ingest tables instead of relying on stale placeholders. The preserved assay-context fields
also keep TI05/TI07 honest about source-specific disagreement and condition-specific host-range behavior instead of
flattening those distinctions away during ingest.

### 2026-03-24: TI05 Harmonize Tier A datasets to the internal canonical schema

#### Executive summary

TI05 now adds a real harmonization boundary under
`lyzortx/pipeline/track_i/steps/build_tier_a_harmonized_pairs.py` instead of letting later Track I steps consume raw
external names directly. The step resolves Tier A bacteria and phage names through the Track A canonical ID maps plus
alias tables, preserves the raw source names for provenance, and marks each external row as either
`overlap_internal_panel` or `novel_to_internal_panel`. A live run on 2026-03-24 produced 68,797 harmonized rows, with
26,029 rows / pairs joinable on canonical internal `pair_id` values and 42,768 rows (38,919 unique pairs) novel to
the current Track A panel.

#### What changed

- Added a dedicated TI05 step that reads all four TI03/TI04 Tier A outputs, requires the Track A ID-map and alias-map
  artifacts, and fails loudly if any prerequisite file is missing.
- Harmonization now rewrites `pair_id`, `bacteria`, and `phage` onto canonical internal names when Track A can resolve
  them, while preserving the original source fields as `source_pair_id_raw`, `source_bacteria_raw`, and
  `source_phage_raw`.
- The step emits `bacteria_id`, `phage_id`, detailed resolution-status fields, and explicit panel-membership flags so
  downstream confidence-tier and cohort steps can distinguish joinable rows from novel or unresolved rows without
  guessing.
- Updated `lyzortx/pipeline/track_i/run_track_i.py` so Track I can run TI05 directly via
  `--step tier-a-harmonization` and so `--step all` executes TI05 between Tier A ingest and later external-label
  processing.
- Updated TI07's default Tier A input path to
  `lyzortx/generated_outputs/track_i/tier_a_harmonization/ti05_tier_a_harmonized_pairs.csv` so later Track I stages
  consume the harmonized canonical schema instead of bypassing TI05 and reading only raw VHRdb output.

#### Findings

- The live TI05 run completed successfully after regenerating Track A and Tier A prerequisites locally. It wrote
  `lyzortx/generated_outputs/track_i/tier_a_harmonization/ti05_tier_a_harmonized_pairs.csv`,
  `ti05_tier_a_harmonization_summary.csv`, and `ti05_tier_a_harmonization_manifest.json`.
- Current overlap vs novelty on the active Track A panel:
  - total harmonized rows: `68,797`
  - total unique external `pair_id` values after harmonization: `64,948`
  - joinable rows / pairs on canonical internal `pair_id`: `26,029 / 26,029`
  - novel rows / pairs outside the internal panel: `42,768 / 38,919`
- Source-by-source overlap is currently highly asymmetric:
  - VHRdb: `26,029` overlap rows / pairs and `30,643` novel rows / pairs
  - BASEL: `468` novel rows / pairs, `0` overlap
  - KlebPhaCol: `7,697` novel rows across `3,848` unique pairs, `0` overlap
  - GPB: `3,960` novel rows / pairs, `0` overlap
- Resolution-status breakdown:
  - `26,029` rows fully resolved on both entities
  - `12,410` rows resolved only on bacteria
  - `219` rows resolved only on phage
  - `30,139` rows unresolved on both entities
- The current Track A build in this checkout produced `405` canonical bacteria and `96` canonical phages. TI05
  therefore computes panel overlap from the actual Track A artifacts rather than hard-coding the older `404x96`
  denominator text from the plan.

#### Interpretation

TI05 makes the external-data seam honest. Later Track I steps no longer have to pretend that raw external strings are
already part of the internal schema: they now receive canonical IDs when resolution exists and explicit novelty flags
when it does not. The live counts also show that Tier A integration is mostly a novelty problem rather than a simple
rename problem, because only VHRdb currently overlaps the active internal panel while BASEL, KlebPhaCol, and GPB are
entirely outside the present Track A alias space.

### 2026-03-24: TI06 Download and ingest Tier B: Virus-Host DB and NCBI BioSample metadata

#### Executive summary

TI06 now performs real network downloads instead of reading imaginary local snapshots. The new step in
`lyzortx/pipeline/track_i/steps/build_tier_b_weak_label_ingest.py` downloads the live Virus-Host DB TSV export, runs a
bounded Entrez `nuccore` search plus `nuccore`/`biosample` `efetch` XML retrievals, and materializes raw downloads plus
normalized CSV outputs under `lyzortx/generated_outputs/track_i/tier_b_weak_label_ingest/`. A live run on 2026-03-24
produced 57,337 Virus-Host DB rows and 268 NCBI Virus/BioSample rows, for 57,605 combined weak-label rows.

#### What changed

- Replaced the invalidated local-file-only TI06 path with actual download helpers for:
  - `https://www.genome.jp/ftp/db/virushostdb/virushostdb.tsv`
  - Entrez `esearch.fcgi` on `db=nuccore`
  - Entrez `efetch.fcgi` on `db=nuccore` and `db=biosample`
- Added raw download outputs under `lyzortx/generated_outputs/track_i/tier_b_weak_label_ingest/raw_ti06_downloads/`:
  `virushostdb.tsv`, `ncbi_nuccore_esearch.json`, `ncbi_nuccore.xml`, `ncbi_virus_report.jsonl`, and
  `ncbi_biosample.xml`.
- Kept the provenance-preserving TI06 CSV contract while making both sources fail fast on empty downloads, empty Entrez
  searches, empty XML responses, and zero-row normalization results.
- Added unit coverage for the new `nuccore` XML parser, raw-download materialization, and the existing BioSample
  attribute parsing path.
- Refreshed `lyzortx/research_notes/external_data/source_registry.csv` so the Tier B rows point at the verified live
  FTP and Entrez endpoints checked on 2026-03-24.

#### Source notes

- Virus-Host DB README:
  `https://www.genome.jp/ftp/db/virushostdb/README`
  Quote: "virushostdb.tsv: Tab separated file containing the following information:"
- Same README:
  `https://www.genome.jp/ftp/db/virushostdb/README`
  Quote: "The host information is collected from RefSeq, GenBank (in free text format), UniProt, ViralZone, and
  manually curated with additional information obtained by literature surveys."
- NCBI Entrez Programming Utilities Help:
  `https://www.ncbi.nlm.nih.gov/books/NBK25499/`
  Quote: "Provides a list of UIDs matching a text query"
- Same docs:
  `https://www.ncbi.nlm.nih.gov/books/NBK25499/`
  Quote: "Returns formatted data records for a list of input UIDs"
- NCBI BioSample attributes:
  `https://www.ncbi.nlm.nih.gov/biosample/docs/attributes/`
  Quote: "`host_disease`"

#### Findings

- The live Virus-Host DB FTP export is stable and machine-readable. Today’s run downloaded `virushostdb.tsv` directly
  from the official GenomeNet FTP listing and normalized it into 57,337 rows with the expected tax ID, lineage, PMID,
  evidence, sample-type, and source-organism provenance fields.
- The old TI06 assumption about a pre-flattened NCBI virus report was wrong. The usable seam is Entrez:
  `esearch` gives the nuccore UID cohort, `efetch` returns the actual `nuccore` XML records, and those records expose
  BioSample cross-references that can be pulled through `efetch` on `db=biosample`.
- The default NCBI query is intentionally bounded:
  `viruses[filter] AND phage[TITL] AND srcdb_refseq[PROP] AND biosample[PROP]`.
  That yielded 268 live rows today, which is enough to satisfy TI06 honestly without turning the step into an
  unbounded Entrez crawl that is likely to be slow or rate-limited in CI.
- Live NCBI QC breakdown on 2026-03-24:
  - `118` rows with `source_qc_flag=ok`
  - `78` rows with `source_qc_flag=host_conflict`
  - `72` rows with `source_qc_flag=biosample_missing`
- BioSample XML parsing is now real rather than hypothetical. The live run preserved non-empty `source_biosample_host_disease`
  on 6 rows. The current query returned 0 non-empty `isolation_host` values, but the parser extracts that attribute
  when present and unit coverage now proves the field is retained.

#### Interpretation

TI06 is now an honest ingestion boundary. Tier B weak labels come from real public downloads, not stale placeholders,
and the code preserves the uncertainty that downstream TI07 needs instead of flattening conflicts or missing BioSample
context away. The bounded Entrez query is the right tradeoff for now: it produces real rows reproducibly and leaves the
door open to broader retrieval later without hard-coding a fragile full-database crawl into the default Track I path.

### 2026-03-24: TI07 Assign confidence tiers to all ingested external labels

#### What was implemented

- Updated `lyzortx/pipeline/track_i/steps/build_external_label_confidence_tiers.py` so TI07 now requires the real TI05
  and TI06 outputs instead of treating external inputs as optional. The step raises `FileNotFoundError` when either
  upstream artifact is missing and raises `ValueError` if any of the six expected sources (`vhrdb`, `basel`,
  `klebphacol`, `gpb`, `virus_host_db`, `ncbi_virus_biosample`) contributes zero tiered rows.
- Kept the four-bin policy small and executable: base confidence still comes from source family
  (`A` assay-backed sources > curated metadata knowledgebase > repository metadata), and row-level degradations still
  come from disagreement, non-`ok` QC flags, and unresolved canonical mapping.
- Expanded the output contract so the TI07 CSV now includes the generic columns required by the plan
  (`confidence_tier`, `training_weight`) while preserving the existing `external_label_*` aliases for downstream Track I
  consumers.
- Added TI07 regression coverage for the new happy path and for the fail-fast case where one expected source is absent
  from the tiered output.

#### Findings

- A live rerun on 2026-03-24 after regenerating Track A plus TI03-TI06 produced
  `lyzortx/generated_outputs/track_i/external_label_confidence_tiers/ti07_external_label_confidence_pairs.csv` with
  `126,402` tiered rows. Overall tier counts were `26,029` `high`, `41,647` `medium`, `58,458` `low`, and `268`
  `exclude`.
- Tier distribution by source on that run:
  - `vhrdb`: `26,029` `high`, `30,643` `medium`, `0` `low`, `0` `exclude`
  - `basel`: `0` `high`, `468` `medium`, `0` `low`, `0` `exclude`
  - `klebphacol`: `0` `high`, `6,576` `medium`, `1,121` `low`, `0` `exclude`
  - `gpb`: `0` `high`, `3,960` `medium`, `0` `low`, `0` `exclude`
  - `virus_host_db`: `0` `high`, `0` `medium`, `57,337` `low`, `0` `exclude`
  - `ncbi_virus_biosample`: `0` `high`, `0` `medium`, `0` `low`, `268` `exclude`
- Unresolved canonical mapping is the dominant downgrade path. All `BASEL`, `GPB`, and `Virus-Host DB` rows were
  penalized by `unresolved_entity_mapping`, and `30,643` of `56,672` `VHRdb` rows were also demoted from `high` to
  `medium` for the same reason.
- `KlebPhaCol` is the only source that currently lands in both `medium` and `low`: `6,576` rows were downgraded only by
  unresolved mapping, while `1,121` rows were pushed further down by the combination of unresolved mapping and
  `source_disagreement`.
- The only `exclude` rows came from `NCBI Virus/BioSample`. Those rows were already weak metadata labels, and the extra
  QC penalties (`host_conflict` or `biosample_missing`) were enough to drive all `268` rows to zero weight.

#### Interpretation

- TI07 is now an honest policy boundary rather than a permissive pass-through. It consumes the real TI03-TI06 outputs,
  proves that every expected source contributed rows, and emits a training-ready table with explicit keep/down-weight/
  exclude semantics.
- The current confidence distribution also exposes the main structural bottleneck for Track I: canonical mapping, not
  assay provenance, is what suppresses trust for most external rows. That means TI08 should treat the external-weighting
  contract as real signal, not as a cosmetic annotation layered on top of uniformly trustworthy data.

### 2026-03-22: TI08 External data as a non-blocking enhancer

#### Executive summary

TI08 now turns the TI07 confidence output into an explicit training-cohort layer rather than forcing external data into
the baseline path. The new step under `lyzortx/pipeline/track_i/steps/build_external_training_cohorts.py` always emits
an `internal_only` arm from ST0.2, then stages optional external additions in the planned order
`+VHRdb -> +BASEL -> +KlebPhaCol -> +GPB -> +Tier B`. When the TI07 artifact is absent, the step still succeeds and
produces a runnable internal-only cohort plus zero-lift summaries for the future external arms.

#### Findings

- The right TI08 seam was dataset assembly, not estimator surgery. Rewriting baseline trainers to consume partial
  external supervision now would have coupled baseline reproducibility to still-evolving source harmonization and
  ablation policy.
- TI07 already provides the exact integration boundary needed for safe enhancement: every external row arrives with a
  source ID, confidence tier, include/exclude decision, and training weight.
- Preserving a `first_training_arm` assignment per row is cleaner than materializing six separate training tables. It
  keeps provenance intact, makes TI09's ordered ablations mechanical, and avoids duplicating the same row into many
  output files.

#### Interpretation

- Track I now has a non-blocking external integration layer: internal-only remains the default runnable baseline, while
  approved external labels can be layered in deterministically when they are present.
- This is the correct amount of implementation for TI08. It makes the enhancement path executable and testable without
  prematurely claiming that the current model trainers should already consume every external row.

### 2026-03-24: TI08 Build training cohorts from internal + external rows

#### Executive summary

TI08 now emits a single cohort file that contains the internal ST0.2 rows plus the TI07 external rows, and it fails
fast if the external side is missing or if every external row is excluded by the confidence policy. The updated
`lyzortx/pipeline/track_i/steps/build_external_training_cohorts.py` contract preserves the legacy
`external_label_*` fields for Track K while also surfacing the generic `confidence_tier` and `training_weight`
columns required by the plan.

#### What changed

- Required TI08 external inputs now include the real TI07 CSV columns, not an optional empty fallback.
- Internal rows are still emitted with `first_training_arm=internal_only`, so the baseline arm remains extractable from
  the same file.
- External rows now carry both the generic and legacy confidence/weight fields in the TI08 cohort output.
- TI08 raises `ValueError` when no external row survives `external_label_include_in_training=1`.
- Added regression coverage for the happy path and the all-excluded failure case.

#### Findings

- The old missing-external shortcut was incompatible with the task acceptance criteria because it allowed a valid TI08
  run to produce an internal-only file.
- The legacy `external_label_*` fields still need to remain in the cohort file because Track K and TI10 already consume
  them directly.
- The new generic `confidence_tier` and `training_weight` columns can coexist with those aliases without changing the
  downstream join contract.

#### Interpretation

TI08 is now an actual integration boundary rather than a permissive passthrough. The cohort file can support both the
internal-only baseline and the external-enhanced arms, but it no longer pretends that an empty or fully excluded
external feed is an acceptable final artifact.

### 2026-03-24: TI09 Strict ablation sequence

#### Executive summary

TI09 now has a dedicated Track I step that loads the TI08 cohort, rebuilds the locked v1 feature space, and retrains
the model arm by arm in the planned strict order `internal-only -> +VHRdb -> +BASEL -> +KlebPhaCol -> +GPB ->
+Tier B`. The step now computes holdout ROC-AUC, top-3 hit rate, and Brier score per arm and raises `ValueError` as
soon as an added source contributes zero external training rows.

#### Findings

- The live end-to-end run now reaches TI09, but it stops at `+BASEL` because the locked feature grid has no joinable
  Basel rows for the current TI08 cohort. That is the correct failure mode for this task: the step refuses to invent a
  Basel ablation result when the added source contributes zero trainable rows.
- The guard is doing something useful here, not just being conservative. The TI08 cohort does contain Basel rows, but
  the TI09 join against the locked ST03 feature table leaves no Basel rows eligible for retraining, so a silent pass
  would have produced a misleading ablation report.
- The implementation still keeps the planned source order explicit and auditable, so if the Basel seam changes in a
  later data refresh, the same step will immediately surface it by completing the full metric table instead of failing.

#### Interpretation

TI09 is now honest. It retrains the locked model per cumulative arm, but it will not manufacture metrics when an added
source cannot actually join onto the model feature grid. In this workspace, Basel is the blocker, so the correct
action is to fail rather than report fake lift.

### 2026-03-22: TI10 Incremental lift and failure modes

#### Executive summary

TI10 now adds `lyzortx/pipeline/track_i/steps/build_incremental_lift_failure_analysis.py`, which reads the TI08
training cohort and TI09 strict-ablation summary to quantify incremental lift at each source-addition step and to
separate external rows by datasource and confidence tier. The step writes three traceable outputs under
`lyzortx/generated_outputs/track_i/incremental_lift_failure_analysis/`: an arm-level lift summary, a source/tier lift
summary, and a failure-mode summary.

#### Findings

- The arm-level lift view is still the cleanest way to express incremental gain: cumulative rows, pairs, and training
  weight all rise monotonically across the strict ablation sequence, while the new deltas make the stepwise lift
  explicit instead of implicit.
- Datasource and confidence-tier slices are better interpreted as trust strata than as model metrics. High-confidence
  VHRdb rows stay in the retained training path, while lower-confidence or unresolved rows are surfaced separately so
  their noise burden is visible.
- Failure modes now resolve into a small set of repeatable buckets: confidence-based exclusion, source disagreement,
  non-`ok` QC states, and unresolved entity mapping. That makes it easier to tell whether a datasource is limited by
  raw label quality or by downstream normalization problems.

#### Interpretation

TI10 is a reporting layer, not a new modeling policy. It makes the progression from TI08 to TI09 auditable at the arm
level and gives the notebook a concrete place to document where lift comes from and which datasource/tier combinations
still fail for reasons that are mechanistic rather than model-specific.

### 2026-03-24: TI03-TI10 invalidated — no external data was ever downloaded

#### Executive summary

Post-merge review found that no Track I step downloads external data. The entire TI03-TI10 chain reads from local paths
that were never populated. All tasks ran on zero rows and reported zero results. TI03-TI10 have been set back to pending
with acceptance criteria requiring actual data downloads and >0 output rows.

#### Evidence

- `grep` for `download|fetch|request|urllib|http` across `lyzortx/pipeline/track_i/steps/` returned zero hits
- `lyzortx/generated_outputs/track_i/` does not exist on disk
- Running `python -m lyzortx.pipeline.track_i.run_track_i --step all` fails with
  `ValueError: No Tier B input files were found.` — the only step that fails fast correctly
- TI08 cohort artifact (consumed by Track K) was never produced

#### What changed

- TI03-TI06: now require downloading real data from source URLs
- TI07-TI10: now require >0 real external rows at each stage
- TI03-TI07 upgraded to gpt-5.4 for external service integration

### 2026-03-24: TI10 source-tier verdicts and failure buckets are explicit

#### What changed

- `lyzortx/pipeline/track_i/steps/build_incremental_lift_failure_analysis.py` now records per-source/tier lift deltas
  against the TI09 internal-only baseline and emits a `lift_direction` classification of `helped`, `hurt`, or
  `neutral`.
- The TI10 manifest now counts only actual failure buckets in `failure_modes` and keeps `clean_row_count` separate so
  the notebook can report real failure modes without mixing them with the non-failure remainder.
- Regression coverage now checks the new verdict column and the failure-mode split so the summary contract stays stable.

#### Interpretation

TI10 is now explicit about what the TI09 numbers mean. The source/tier table is no longer just a pile of deltas: it
has a direct helped/hurt/neutral verdict for each source+tier slice, and the manifest cleanly separates true failure
states from the rows that simply passed all checks.
