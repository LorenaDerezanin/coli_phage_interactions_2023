# FNA

The assembled genome of each host strain in the collection (34 in total).

# LPS_type

Outer core LPS typing of the strains based on the waaL detection (using Blast and a K12 waaL as a query) and clustering
(using MMSeqs2).

# panacota

The Panacota analysis of the collection : core genome phylogeny (tree .nkw), proteins, replicons, etc.

# ClermontTyping

Clermont typing to obtain the _phylogroup_ of each strain using ClermonTyper (the git repo is available in my
30_dev/trash directory). Command : `./clermonTyping --fastafile host_list.txt` (absolute paths in the host list,
obtained with the `readlink -f ..../FNA/*` command)

# O-typing (and H-typing)

O-antigen and H-typing of each strain obtained using ECTyper (https://github.com/phac-nml/ecoli_serotyping, installed in
my WSL `phd` conda environment). See log file for more information.

# ST

MLST was performed in silico using the mlst package (https://github.com/tseemann/mlst) with the `ecoli_achtman_4` scheme
(not the `ecoli` scheme) using the following bash command :
`./bin/mlst --scheme ecoli_achtman_4 --legacy /mnt/d/These/20_data/201_genomic_data/370_and_host/host/FNA/* > host_strains_ST.tsv`

The output was then formatted (to add original strain name).
