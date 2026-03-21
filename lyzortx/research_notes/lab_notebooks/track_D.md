### 2026-03-21: TD01 phage sequence processing pipeline

#### What was implemented

Added `lyzortx/pipeline/track_d/run_track_d.py` and
`lyzortx/pipeline/track_d/steps/build_phage_protein_sets.py` to build reproducible per-phage protein FASTA sets from
the phage genomics directory in one command. The pipeline resolves the canonical 96-phage panel from
`data/genomics/phages/guelin_collection.csv`, prefers supplied protein FASTA when available, falls back to GenBank CDS
translations when present, and otherwise calls proteins from genome FASTA/GenBank sequence with `pyrodigal`.

#### Output summary

Outputs are written under `lyzortx/generated_outputs/track_d/phage_protein_sets/`:

- `protein_fastas/<phage>.faa`: per-phage protein sets
- `phage_protein_summary.csv`: one row per phage with input provenance and protein counts
- `manifest.json`: regeneration command, gene-calling policy, and ignored non-panel inputs

The current repo data contains 96 panel phages in `guelin_collection.csv` and 97 genome FASTAs under `FNA/`; the extra
genome file is `411_P3.fna`, which is outside the current panel and is therefore reported but not processed.

#### Interpretation

This closes the immediate gap between raw phage genomes and downstream protein-level feature work. Track D and Track E
can now consume canonical per-phage protein sets without depending on ad hoc notebook code or external binaries.

#### Next steps

Use the generated protein FASTAs as the shared protein input for RBP/domain annotation and then layer the planned
feature blocks on top of the stable per-phage identifiers emitted here.
