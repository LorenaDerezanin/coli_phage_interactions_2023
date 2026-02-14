# Lyzor Tx

This is a forked repository. While the original intent was reproducing the
analysis of the paper "[Prediction of strain level phage-host interactions
across the Escherichia genus using only genomic
information](https://doi.org/10.1038/s41564-024-01832-5)", the new aim is
to produce an alternative pipeline based on the raw input data of the 
research paper. The new pipeline will be developed in the [lyzortx](.)
directory in order not to mix with the sources of the original repository.

## Getting started
See [INSTALL.md](../INSTALL.md) for detailed instructions on how to set
up your development environment and run the pipelines.

## Research notes
We are recording detailed research notes in the [research_notes](research_notes)
directory.

## Original paper pipeline reproduction steps
- Run `predict_all_phages.py` without any arguments
  - This generates different outputs than what was originally committed in `dev/predictions/results/(logs|performances)`
- Section 2 from the README can not be reproduced because the 
  `recommend_cocktail.py` script the README refers to does not exist.
