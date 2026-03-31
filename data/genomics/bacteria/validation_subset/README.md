# Track L raw-host validation subset

This directory contains a tiny committed host-assembly subset used to validate raw-input plumbing for Track L host-side
projectors.

Why this exists:

- The full Picard collection assemblies are published separately on figshare and are too large to vendor wholesale.
- `TL16` needed a stable, reviewable raw-host input subset so future work does not depend on ad hoc one-off downloads.
- The files here are validation fixtures only, not a replacement for the full collection.

Source dataset:

- Florian Tesson (2024). *Genome assembly of the Escherichia Picard collection*. figshare. Dataset.
- DOI: `10.6084/m9.figshare.25941691.v1`
- Landing page: <https://figshare.com/articles/dataset/Genome_assembly_of_the_Escherichia_Picard_collection/25941691>
- License: `CC BY 4.0`

Included hosts:

- `EDL933`
- `LF82`
- `55989`

Selection rationale:

- `EDL933` and `LF82` are already important Track L round-trip hosts.
- `55989` adds a third panel host with a different serotype / capsule profile.
- Together they provide a small but nontrivial raw-input subset for future projector validation.

Integrity and attribution:

- Exact file metadata, figshare file IDs, source URLs, retrieval date, and local checksums are recorded in
  `manifest.json`.
- Any redistribution of these files must preserve attribution under the upstream `CC BY 4.0` terms.
