# Track D Sequence Processing

`python lyzortx/pipeline/track_d/run_track_d.py`

This command scans `data/genomics/phages/`, resolves the canonical 96-phage panel from
`data/genomics/phages/guelin_collection.csv`, and runs the implemented Track D sequence-derived feature builders:

1. `protein-sets`: write reproducible per-phage protein FASTA files to
   `lyzortx/generated_outputs/track_d/phage_protein_sets/protein_fastas/`
2. `genome-kmers`: write tetranucleotide SVD features to
   `lyzortx/generated_outputs/track_d/phage_genome_kmer_features/phage_genome_kmer_features.csv`
3. `viridic-distance`: write VIRIDIC tree-distance MDS features to
   `lyzortx/generated_outputs/track_d/phage_distance_embedding/phage_distance_embedding_features.csv`

Input precedence is deterministic:

1. Existing protein FASTA (`.faa`, `.aa`, `.pep`)
2. GenBank (`.gb`, `.gbk`, `.genbank`) CDS `/translation=` qualifiers
3. Genome FASTA (`.fna`, `.fa`, `.fasta`) or GenBank `ORIGIN` sequence with `pyrodigal`

The genome k-mer builder fits a deterministic TruncatedSVD embedding on normalized tetranucleotide frequencies from all
discovered genome FASTA files in `FNA/`, then emits panel-joinable rows with `phage_gc_content`,
`phage_genome_length_nt`, and `phage_genome_tetra_svd_00..23`. Any non-panel genomes remain documented in the source
summary and manifest but are excluded from the joinable feature CSV.

The VIRIDIC distance builder parses
`data/genomics/phages/tree/96_viridic_distance_phylogenetic_tree_algo=upgma.nwk`, computes leaf-to-leaf patristic
distances, and fits a deterministic metric MDS embedding with `phage_viridic_mds_00..07` coordinates. It also writes
an explicit pairwise-distance CSV for auditability alongside the feature CSV, metadata CSV, and manifest.

Tracked documentation for the output formats lives in each step's `manifest.json` plus the corresponding summary/metadata
CSV files under the output directories.
