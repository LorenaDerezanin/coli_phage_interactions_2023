# AUTORESEARCH Program Contract

## Scope

AR02 freezes the AUTORESEARCH sandbox before feature-family implementation. The fixed surface is:

- `lyzortx/autoresearch/prepare.py`
- `lyzortx/autoresearch/train.py`
- `lyzortx/autoresearch/README.md`
- `lyzortx/autoresearch/program.md`

`prepare.py` owns the one-time cache build. `train.py` owns the repeated short experiment loop.

## Frozen cache layout

The default output root is `lyzortx/generated_outputs/autoresearch/`.

- `ar01_canonical_pair_table_v1.csv`
- `ar01_split_benchmark_manifest_v1.json`
- `ar01_input_checksums_v1.json`
- `ar01_label_policy_v1.json`
- `search_cache_v1/ar02_search_cache_manifest_v1.json`
- `search_cache_v1/ar02_schema_manifest_v1.json`
- `search_cache_v1/ar02_provenance_manifest_v1.json`
- `search_cache_v1/search_pairs/train_pairs.csv`
- `search_cache_v1/search_pairs/inner_val_pairs.csv`
- `search_cache_v1/feature_slots/<slot>/entity_index.csv`
- `search_cache_v1/feature_slots/<slot>/features.csv` (when a slot has been materialized)
- `search_cache_v1/feature_slots/<slot>/schema_manifest.json`

## Frozen slot contract

The slot names, join keys, and namespace prefixes are fixed in AR02:

| Slot | Join key | Prefix |
| --- | --- | --- |
| `host_defense` | `bacteria` | `host_defense__` |
| `host_surface` | `bacteria` | `host_surface__` |
| `host_typing` | `bacteria` | `host_typing__` |
| `host_stats` | `bacteria` | `host_stats__` |
| `phage_projection` | `phage` | `phage_projection__` |
| `phage_stats` | `phage` | `phage_stats__` |

Future tasks may add columns inside these slots, but they may not rename a slot, change its join key, or change its
column-family prefix.

## Implemented slot behavior

- `host_surface` is now materialized by `prepare.py` from raw host FASTAs using the pyhmmer fast path recorded in the
  DEPLOY notebook. The exported columns stay inside the `host_surface__` namespace and intentionally exclude
  `host_lps_core_type`, which still depends on Picard lookup tables rather than raw-sequence evidence.
- The host-surface build caches `predicted_proteins.faa` under
  `lyzortx/generated_outputs/autoresearch/host_surface_cache_build/` so retries do not rerun the front half of the
  pipeline.

## Search-space boundary

- The search cache contains `train` and `inner_val` only.
- Sealed holdout labels and holdout-ready evaluation tables never enter the search workspace.
- Warm caches are optional accelerators only and must declare the same `schema_manifest_id` plus the same slot
  contract.
