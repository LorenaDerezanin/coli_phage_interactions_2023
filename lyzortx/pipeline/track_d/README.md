# Track D Sequence Processing

`python lyzortx/pipeline/track_d/run_track_d.py`

This command scans `data/genomics/phages/`, resolves the canonical 96-phage panel from
`data/genomics/phages/guelin_collection.csv`, and writes reproducible per-phage protein FASTA files to
`lyzortx/generated_outputs/track_d/phage_protein_sets/protein_fastas/`.

Input precedence is deterministic:

1. Existing protein FASTA (`.faa`, `.aa`, `.pep`)
2. GenBank (`.gb`, `.gbk`, `.genbank`) CDS `/translation=` qualifiers
3. Genome FASTA (`.fna`, `.fa`, `.fasta`) or GenBank `ORIGIN` sequence with `pyrodigal`

Tracked documentation for the output format lives in `manifest.json` and `phage_protein_summary.csv` under the output
directory. The manifest records the exact one-command regeneration path, ignored non-panel inputs, and the gene-calling
policy used when proteins are derived from genomes.
