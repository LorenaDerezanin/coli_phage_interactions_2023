# coli_phage_interactions_2023
Git repository containing the code and genomics data published in the article Gaborieau, Vaysset, Tesson et al. (2025), *Predicting phage-bacteria interactions at the strain level from genomes*.

# Repository
- data:
    - data/genomics: All genomics input data in regarding phages and bacteria that are used in all the article.
	- data/genomics/bacteria: All genomics data regarding the Picard collection of bacteria (See `picard_collection.csv` for a summary of bacteria-related input features). The tools and intermediate results used to generate the `picard_collection.csv` file are presented in the subdirectories (capsules, antiphage systems, O-type, H-type, outer core LPS type, outer membrane proteins clustered, core genome phylogeny using PanACoTA (`panacota`) and UMAP projection of the phylogenetic distances in 8-dimensional vectors (`umap_phylogeny`)).
		- `data/genomics/bacteria/isolation_strains`: Detailed analysis on the strains that were used to isolate phages present in the Guelin collection of phages. 
    	- data/genomics/phages: All genomics data regarding the Guelin collection of phages (See `guelin_collection.csv` for a summary of phage-related features).
		- data/genomics/phages/FNA: Nucleic acid sequences of the genome of each phage in the collection (fasta format). 
		- data/genomics/phages/RBP: Results of the detection of phage Receptor-Binding Proteins (RBP).
		- data/genomics/phages/tree: Tree of the phages (Newick format).	
    - data/interactions: The phage-bacteria interaction matrix between the Picard collection of *Escherichia* strains and the Guelin collection of phages (`interaction_matrix.csv`) and the Jupyter notebook to visualize it interactively using Bokeh (`bokeh_draw_interaction_matrix.ipynb`).
- dev:
    - dev/cocktails: All the data (`data`) and the analysis notebook (`analyze_cocktail_interactions_tailored_vs_generic_vs_baseline.ipynb`) of the cocktails experiment (Figure 6)
    - dev/inference_bacteria/phage_per_phage: All the data and statistical analysis (linear-mixed models, LMM) presented in Figure 4. Statistical inference is performed in a phage-per-phage manner (i.e. one statistical model per phage). 
    - dev/predictions: All the data and predictive models predicting phage-bacteria interactions in a pointwise manner (Figure 5).  
	- dev/predictions/per_phage_perf.csv: Summary of the performance of each phage-specific predictive model.
	- dev/predictions/results: For each phage, performance of all tested models on 10-Fold Cross-Validation (`performance`), logs (`logs`), feature importance (`feature_importances.rar`) and all predictions (`predictions.rar`).

# Phage–Host Interaction Prediction Pipelines

The rest of this document describes in more details the two complementary bioinformatics workflows developed to (i) train predictive models of phage–bacterium interactions (Figure 5) and (ii) recommend optimal phage cocktails for *E. coli* strains (Figure 6), as published in **Gaborieau et al. (2024), *Prediction of strain-level phage–host interactions across the Escherichia genus using only genomic information*, Nature Microbiology**.

It provides a step-by-step description of how both workflows can be implemented in Python using publicly available tools.
The goal is to enable researchers to reproduce or adapt the pipelines for their own datasets.

---

## 1. Predictive Model Training Pipeline

### 1.1 Overview

This pipeline aims to train predictive models for each phage in the Guelin collection to estimate the probability of a lytic interaction with any *E. coli* strain.
A distinct model is fitted for each phage using bacterial genomic features as inputs (no phage features).

The process involves five main steps:

1. Genome annotation and core genome phylogeny
2. Extraction of genomic features
3. Feature encoding and normalization
4. Model training and cross-validation
5. Model selection and evaluation

---

### 1.2 Input Data

* **FASTA genome assemblies** of the bacterial strains (available [here](https://figshare.com/articles/dataset/Genome_assembly_of_the_Escherichia_Picard_collection/25941691/1))
* **Interaction matrix** (CSV) describing known lytic/non-lytic phage–bacterium pairs between phages in the Guelin collection and bacterial strains in the Picard collection (available [here](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/interactions/interaction_matrix.csv))
* **Metadata** (strain identifiers, phage names, and phylogenetic clusters for cross-validation)

Expected directory structure for the pipeline:

```
data/
├── genomes/
├── interactions/interaction_matrix.csv
├── metadata/
└── features/
```

---

### 1.3 Genome Annotation and Core Genome Phylogeny

Annotation and phylogenetic reconstruction are performed using [`PanACoTA`](https://github.com/gem-pasteur/PanACoTA), which annotates genomes, computes the core genome, and builds a phylogenetic tree.

Two categories of genomic information are extracted:

* **Gene annotation** – required for identifying macromolecular systems, which is then performed using [`MacSyFinder`](https://github.com/gem-pasteur/macsyfinder), a tool for detecting gene clusters encoding macromolecular systems based on (i) homology of individual components and (ii) co-localization on the replicon.
* **Core genome phylogeny** – used to encode phylogenetic distances. The phylogenetic tree is available [here](https://github.com/mdmparis/coli_phage_interactions_2023/tree/main/data/genomics/bacteria/panacota/tree). Once the phylogenetic tree is built, it is converted into a pairwise phylogenetic distance matrix using an in-house script. This matrix is then reduced into an 8-dimensional coordinate system using the UMAP algorithm (Python module). This step is essential to encode the phylogenetic information into a format suitable for machine learning algorithms. At the end of this step, each bacterial strain’s phylogenetic position is encoded by an 8-dimensional vector.
The UMAP coordinates are available [here](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv).

> **Note:** Pairwise phylogenetic distances between strains are also used to group strains when building training and evaluation sets. Strains with a core genome distance <10⁻⁴ substitutions per site are always placed in the same fold for cross-validation.

---

### 1.4 Extraction of Genomic Features

| Feature                             | Tool / Source                                                                                                                                                                                                  | Description                                                                                               |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Phylogroup**                      | [`ClermonTyping`](https://github.com/A-BN/ClermonTyping)                                                                                                                                                       | In silico *E. coli* phylogroup typing                                                                     |
| **Sequence Type (ST)**              | [`SRST2`](https://github.com/katholt/srst2)                                                                                                                                                                    | MLST typing                                                                                               |
| **O-antigen & H-antigen serotypes** | [`ECTyper`](https://github.com/phac-nml/ecoli_serotyping)                                                                                                                                                      | Serotype prediction                                                                                       |
| **Outer core LPS type**             | Homology detection of *waaL* using `blastp`, followed by phylogenetic clustering of *waaL* clades                                                                                                              | Manual phylogenetic classification; could be automated by blasting representative sequences for each type |
| **ABC-dependent capsule serotype**  | [`MacSyFinder`](https://github.com/gem-pasteur/macsyfinder) custom models ([repo link](https://github.com/mdmparis/coli_phage_interactions_2023/tree/main/data/genomics/bacteria/capsules/ABC_capsules_types)) | CapsuleFinder + custom serotyping                                                                         |
| **Klebsiella-like capsule**         | [`Kaptive`](https://github.com/katholt/Kaptive)                                                                                                                                                                | Capsule presence and serotype detection                                                                   |

All feature data are available in [this folder](https://github.com/mdmparis/coli_phage_interactions_2023/tree/main/data/genomics/bacteria) and summarized in [picard_collection.csv](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/genomics/bacteria/picard_collection.csv).

The final feature matrix includes:

```
UMAP_1 ... UMAP_8
O_serotype
LPS_outer_core
ST
ABC_capsule_serotype
Klebsiella_capsule_serotype
```

---

### 1.5 Feature Encoding

Categorical features are one-hot encoded; numerical ones are standardized (centered to mean 0 and scaled to unit variance).
Rare categories (<3 occurrences) are grouped together.

Example (O-antigen encoding):

```
O1: 1, O2: 0, O3: 0 ...
```

---

### 1.6 Model Training

For each phage, four models are trained using `scikit-learn` v1.1.2:

1. Logistic regression (L2 penalty)
2. Logistic regression (L1 penalty)
3. Random forest (max_depth=3, n_estimators=250)
4. Random forest (max_depth=6, n_estimators=250)

**Inputs:**
Interaction matrix + bacterial features (`UMAP`, `Outer core LPS type`, `ST`, `Klebsiella capsule serotype`, `O-antigen`, `ABC capsule serotype`), extracted from
[picard_collection.csv](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/genomics/bacteria/picard_collection.csv) and
[coli_umap_8_dims.tsv](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv).

**Class imbalance correction:**
To prevent models from always predicting the majority (negative) class (81% of phage–bacterium pairs are negative, 19% positive), class weights are adjusted:

| % Positive Interactions | Positive Class Weight |
| ----------------------- | --------------------- |
| >60%                    | 0.8                   |
| 40–60%                  | 1                     |
| 30–40%                  | 1.5                   |
| 20–30%                  | 2                     |
| <20%                    | 3                     |

**Cross-validation:**
Group 10-fold cross-validation based on core genome distance (<10⁻⁴ threshold): strains with a core genome distance <10⁻⁴ substitutions per site are always placed in the same fold for cross-validation.
This ensures each point in the dataset is used in at least one test fold, and similar strains are always grouped together.
Cluster file (370+host_cross_validation_groups_1e-4.csv) is not available anymore.

---

### 1.7 Model Evaluation and Selection

**Metrics:**
Precision, Recall, F1, AUROC, and Average Precision (AUPR).

All results for the 100 models per phage are available [here](https://github.com/mdmparis/coli_phage_interactions_2023/tree/main/dev/predictions/results):

* `/performance`: contains performance metrics including average precision for each model
* `/predictions`: contains the predicted probability of infection for each phage–strain pair

The best model for each phage is selected based on the mean AUPR across folds.
Example: `RF_250_6_weight=2; fold=9` was selected for phage T4LD.

**Outputs:**

```
models/best_model_<phage>.pkl
results/performance_<phage>.csv
results/predictions_<phage>.csv
```

Example GitHub resources:

* [interaction_matrix.csv](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/interactions/interaction_matrix.csv)
* [performance_T4LD_Group10Fold_CV.csv](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/dev/predictions/results/performances/performance_T4LD_Group10Fold_CV.csv)

---

### 1.8 Example Command

The script used in the paper is available [here](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/dev/predictions/predict_all_phages.py).
Please note that not all input files used in the publication are publicly available.

```
python predict_all_phages.py \
    --matrix data/interactions/interaction_matrix.csv \
    --features data/features/bacterial_features.csv \
    --cv data/metadata/cv_clusters.csv \
    --outdir results/
```

---

## 2. Phage Cocktail Recommendation Algorithm

### 2.1 Overview

This algorithm uses the previously trained models to recommend an optimal combination of up to three phages for any given *E. coli* strain.
It consists of four successive decision layers, ordered by decreasing specificity.
Each step ranks phages according to the predicted infection probability and retains those with a score >0.5.
The upstream steps have priority, as they are empirically more precise.

---

### 2.2 Input Data

* Trained model files (`.pkl`)
* Interaction matrix
* Bacterial feature table (same structure as used for model training)

---

### 2.3 Step 1 — “Same as Host”

For each phage, check whether the target strain shares key features with the phage’s isolation strain (host strain):

| Feature checked             | Comparison     |
| --------------------------- | -------------- |
| O-antigen                   | same/different |
| Sequence type (ST)          | same/different |
| ABC capsule serotype        | same/different |
| Klebsiella capsule serotype | same/different |
| LPS outer core type         | same/different |

A random forest model (depth=4, 80 estimators, positive class weight=1.5) is used to predict infection probability based on these features only.

---

### 2.4 Step 2 — “Similar to a Strain of the Picard Collection”

This step does not use machine learning.
A k-nearest-neighbour-like approach identifies strains in the Picard collection sharing the same haplotype (ST + O + H antigens) as the query strain.
Phages infecting at least half (≥50%) of these strains are recommended, ranked by their coverage.

---

### 2.5 Step 3 — “Trait Matching”

If fewer than three phages have been selected, this step applies the previously trained machine learning models (random forests, depth=4, 80 estimators, positive class weight=1.5) using the complete feature set (UMAP, Outer core LPS type, ST, Klebsiella capsule serotype, O antigen, ABC capsule serotype).
Phages with predicted infection probability >0.5 are recommended, ranked by probability.

---

### 2.6 Step 4 — “Rescue”

If fewer than three phages are still recommended, the remaining slots are filled with the most **generalist phages** from the Guelin collection (those with the highest overall lytic coverage).

To preserve diversity:

* Two phages from the same genus or isolated on the same host cannot both be recommended.

---

### 2.7 Output

Results from the publication are available [here](https://github.com/mdmparis/coli_phage_interactions_2023/tree/main/dev/cocktails).

The algorithm returns:

```
results/recommendations_<strain>.csv
```

with columns:

```
phage_id, predicted_probability, decision_step
```

---

### 2.8 Example Command

```
python recommend_cocktail.py \
    --features data/features/query_strain.csv \
    --models models/ \
    --matrix data/interactions/interaction_matrix.csv \
    --outdir results/
```

---

## 3. Global Workflow Diagram

```mermaid
flowchart TD
    A[Genome assemblies (FASTA)] --> B[Annotation & core genome phylogeny (PanACoTA)]
    B --> C[Feature extraction (MacSyFinder, SRST2, ECTyper...)]
    C --> D[Feature encoding (UMAP, one-hot)]
    D --> E[Model training (per phage)]
    E --> F[Model selection (best AUPR)]
    F --> G[Trained models (.pkl)]
    G --> H[Phage cocktail recommendation algorithm]
    H --> I[Phage recommendations (CSV)]
```

---

## 4. Reproducibility

* Python 3.9+
* `scikit-learn` 1.1.2
* `umap-learn`, `pandas`, `numpy`, `matplotlib`
* Compatible with Conda or Docker environments

Example environment setup:

```
conda env create -f environment.yml
conda activate phage-predict
```

---

## 5. References

* PanACoTA: [https://github.com/gem-pasteur/PanACoTA](https://github.com/gem-pasteur/PanACoTA)
* MacSyFinder: [https://github.com/gem-pasteur/macsyfinder](https://github.com/gem-pasteur/macsyfinder)
* SRST2: [https://github.com/katholt/srst2](https://github.com/katholt/srst2)
* ECTyper: [https://github.com/phac-nml/ecoli_serotyping](https://github.com/phac-nml/ecoli_serotyping)
* Kaptive: [https://github.com/katholt/Kaptive](https://github.com/katholt/Kaptive)
* Data repository: [https://github.com/mdmparis/coli_phage_interactions_2023](https://github.com/mdmparis/coli_phage_interactions_2023)

---

# Getting Started (by @zoltanmaric)
This repository uses `micromamba` to manage environments.
You can install it using the official docs
[here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

Then create and activate the `micromamba` environment using the following commands:
```bash
micromamba create -f environment.yml
micromamba activate phage_env
```

To enable running Jupyter notebooks using this environment, create
a dedicated Jupyter kernel for it:
```bash
python -m ipykernel install --user --name phage_env --display-name phage_env
```

Now when you run notebooks in `jupyter lab`, make sure to select the `phage_env`
kernel.

## Research Notes
Predicted cocktails are benchmarked against a "baseline" and a "generic" cocktail.
From the paper:
> To evaluate the performance of our algorithm, we compared the performance of
> the tailored and generic cocktails with an uninformed baseline cocktail.
> The latter corresponds to the most-covering triplet of phages, that is,
> the cocktail of three phages which together lyse the largest number of
> bacteria in the original interaction matrix. This baseline cocktail yielded
> productive lysis on 63% of the original Picard collection.

> First, some cocktails (n = 15, recommended to 24 test bacteria) contained
> phages recommended at early stages of the pipeline (denoted as ‘tailored’ 
> in the rest of the text), while the rest of the cocktails (n = 3, recommended
> to 76 bacteria) were recommended at late stages of the pipeline (denoted as
> ‘generic’ in the rest of the text) (Fig. 6c).

@zoltanmaric's interpretation: 
- **Baseline**: The single cocktail of 3 phages that together lyse the largest
  number of bacteria (63% of the original Picard collection)
- **Generic**: The top 3 most recommended cocktails by the recommendation pipeline
  built in this paper.
