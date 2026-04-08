# AUTORESEARCH Track

## Provenance

This track is inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — an AI-driven research
automation framework tested on a single NVIDIA H100. Our implementation adapts the concept for phage-host interaction
prediction with a fixed LightGBM baseline, not an autonomous model search agent.

## Sandbox Boundaries

- `prepare.py` is the only path from raw inputs to the search cache.
- `train.py` is the only experiment surface. It consumes a prepared cache and must not rebuild it.
- `replicate.py` handles AR09 candidate import and sealed-holdout replication.
- `program.md` freezes the cache layout, schema contract, provenance metadata, and warm-cache policy.

## GPU Policy

- `train.py` supports both `--device-type gpu` and `--device-type cpu`.
- The current baseline (64-tree LightGBM) is lightweight enough to run on CPU in under 30 minutes.
- RunPod GPU provisioning exists for future heavier workloads and to validate the orchestration pipeline.
- When provisioning RunPod pods, check community GPU availability first — the locked GPU type may be out of stock.
