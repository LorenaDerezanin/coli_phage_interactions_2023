# AUTORESEARCH Sandbox

`lyzortx/autoresearch/` is the fixed sandbox surface for the raw-input AUTORESEARCH track.

- `prepare.py` is the only supported path from raw inputs to the search cache.
- `train.py` is the short-loop experiment surface. It consumes a prepared cache and must not rebuild it.
- `program.md` freezes the cache layout, schema contract, provenance metadata, and warm-cache policy.

## Commands

Build the frozen cache contract from raw inputs:

```bash
micromamba run -n phage_env python lyzortx/autoresearch/prepare.py \
  --skip-host-assembly-resolution
```

Validate or consume an existing cache for short experiments:

```bash
micromamba run -n phage_env python lyzortx/autoresearch/train.py
```

## Output contract

`prepare.py` writes under `lyzortx/generated_outputs/autoresearch/`:

- AR01 raw-input contract artifacts at the output root.
- AR02 search cache artifacts under `search_cache_v1/`.
- Only `train` and `inner_val` pair tables are exported into the search cache.
- Materialized feature slots write `feature_slots/<slot>/features.csv` plus updated slot schema manifests.
- `host_surface` is currently built from raw host FASTAs with the pyhmmer fast path and excludes Picard-derived
  `host_lps_core_type`.
- `host_typing` is currently built from raw host FASTAs via the pinned phylogroup, serotype, and MLST callers, with
  unresolved caller outputs recorded in a slot build manifest instead of being coerced to placeholders.
- `host_stats` is currently built from raw host FASTAs as a low-cost numeric baseline block
  (`record_count`, `genome_length_nt`, `gc_content`, `n50_contig_length_nt`).
- Sealed holdout labels and holdout-ready evaluation tables stay outside the search workspace entirely.

## Warm caches

Checked-in DEPLOY feature CSVs are optional warm-cache accelerators only. They are never source-of-truth inputs for
AUTORESEARCH. If a warm cache is provided, `prepare.py` validates that its manifest matches the frozen AR02 schema
exactly before recording it in provenance.
