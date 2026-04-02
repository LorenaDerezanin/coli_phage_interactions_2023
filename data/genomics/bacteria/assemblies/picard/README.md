# Picard Collection Genome Assemblies

403 *Escherichia* genome assemblies from the Picard collection.

## Source

- **DOI**: [10.6084/m9.figshare.25941691.v1](https://doi.org/10.6084/m9.figshare.25941691.v1)
- **Citation**: Tesson, F. "Genome assembly of the *Escherichia* Picard collection." figshare (2024).
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## License Terms

This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). Redistribution, including in
CI container images, is permitted with attribution. The citation above satisfies the attribution requirement.

## Contents

One FASTA file per strain (403 files). File names match the `bacteria` column in
`data/genomics/bacteria/picard_collection.csv`.

## Usage

These assemblies are the raw input for the Deployment-Paired Feature Pipeline (DEPLOY track). All host genomic
features — defense subtypes, surface receptors, capsule profiles, phylogroup, serotype, MLST — are derived from these
FASTAs at both training time and inference time to ensure exact feature parity.
