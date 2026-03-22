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

### 2026-03-21: TD02 phage genome tetranucleotide embedding feature block

#### What was implemented

Added `lyzortx/pipeline/track_d/steps/build_phage_genome_kmer_features.py` and wired it into
`lyzortx/pipeline/track_d/run_track_d.py` as the `genome-kmers` step. The builder scans
`data/genomics/phages/FNA/`, computes normalized tetranucleotide (`k=4`) frequency vectors per genome FASTA, fits a
deterministic `TruncatedSVD` embedding across all discovered genomes, and emits a panel-joinable feature CSV with 24
embedding coordinates plus GC content and genome length.

#### Output summary

Outputs are written under `lyzortx/generated_outputs/track_d/phage_genome_kmer_features/`:

- `phage_genome_kmer_features.csv`: 96 canonical panel phages with 26 numeric features plus the `phage` key
- `phage_genome_kmer_source_summary.csv`: source-level provenance and genome summary metrics for all 97 discovered FNA
  files
- `phage_genome_kmer_feature_metadata.csv`: column-level transform and provenance metadata
- `manifest.json`: regeneration command, effective SVD dimensionality, explained variance, and non-panel genome list

On the current repo data, the step fits the SVD on all 97 discovered genomes in `FNA/`, emits 96 joinable panel rows,
and retains `411_P3` in the source summary only because it is outside `guelin_collection.csv`. The 24 retained SVD
dimensions capture 99.42% of the tetranucleotide-frequency variance. Across the panel output, genome length ranges from
39,039 nt to 171,847 nt and GC content ranges from 0.353 to 0.546.

#### Interpretation

This closes the genome-composition feature gap for Track D with a compact, fully numeric phage feature block that is
ready to merge into downstream pair tables. Fitting on all discovered genomes preserves the local unsupervised geometry,
while restricting the final feature CSV to canonical panel phages keeps the downstream join contract clean.

#### Next steps

Combine this block with the pending RBP annotations (`TD01`) and VIRIDIC distance embedding (`TD03`), then measure
whether the added phage-side signal improves holdout performance beyond the current identity-heavy baselines.
