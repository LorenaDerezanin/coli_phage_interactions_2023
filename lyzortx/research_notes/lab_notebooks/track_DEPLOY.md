### 2026-04-02: Track rationale — Deployment-Paired Feature Pipeline

#### Executive summary

This track eliminates the training/inference feature mismatch discovered in the TL18 post-merge audit. The model is
currently trained on curated panel metadata but scored at inference time on features derived from raw genome assemblies
by bioinformatics tools. These two paths produce systematically different feature values. This track makes them
identical by re-deriving all training features from raw assemblies using the same pipeline that runs at inference time.
It also replaces binary feature thresholds with continuous scores where the gradient carries biological signal, and
removes ~95 redundant or derived features.

#### Motivation from the TL18 audit

The TL18 audit (2026-04-02, documented in `track_L.md`) walked through every inference step on 3 validation hosts and
found:

- **Training/inference parity bugs**: DefenseFinder version drift (17.3% of model importance affected), capsule HMM
  sensitivity mismatch (3-5% importance), extra phage in FNA directory.
- **Binary thresholds discarding gradients**: 141 of 190 numeric features are binary. Receptor phmmer bit scores, RBP
  family mmseqs percent identity, capsule HMM profile scores, and defense gene counts all carry biological information
  that the current encoding throws away.
- **Feature redundancy**: 91 wasted one-hot features from exact duplicates (host_o_type duplicates host_o_antigen_type,
  etc.) plus 4 derived summary features (defense_diversity, has_crispr, abi_burden, rbp_family_count).

#### Design decisions

**What switches to continuous scores (and why):**

- **Receptors** (12 features): phmmer bit scores replace binary. A receptor at 99% identity is biologically different
  from one at 60% — variant proteins may not function as phage receptors. The score is a proxy for functional integrity.
- **RBP families** (32 features): mmseqs percent identity replaces binary. A phage with a 90% identity match to an RBP
  family is more likely to use that receptor-binding strategy than one at 31%.
- **Capsule** (~30 features): per-profile HMM scores replace binary capsule_present + categorical serotype. The paper's
  CapsuleFinder required syntenic gene co-localization to call a capsule present; our HMM scan detects individual gene
  hits. Rather than reproducing the paper's threshold (which requires a tool we don't ship), we expose the raw scores
  and let the model learn what patterns of capsule gene presence matter. Scattered capsule gene fragments that don't
  form a functional locus will have weak, dispersed scores that the model can learn to discount.
- **Defense subtypes** (79 features): integer gene counts replace binary. Two copies of MazEF is biologically different
  from one — the redundancy makes it harder for a phage to neutralize both simultaneously. The HMM detection *score* is
  a tool artifact (detection confidence, not defense strength), but the *count* is real biology.

**What stays categorical:**

- Phylogroup, serotype, sequence type, O-antigen type, LPS core type — these are genuinely categorical biological
  classifications with no meaningful continuous encoding.

**What gets dropped:**

- `host_o_type` (exact duplicate of `host_o_antigen_type`): 84 wasted one-hot columns
- `host_surface_lps_core_type` (exact duplicate of `host_lps_core_type`): 6 wasted columns
- `host_capsule_abc_present` (exact duplicate of `host_capsule_abc_proxy_present`): 1 wasted column
- `host_o_antigen_present`, `host_lps_core_present` (derivable from whether the categorical is non-empty)
- `host_k_antigen_type_source` (metadata about feature provenance, not biology)
- `host_defense_has_crispr` (max of CRISPR-Cas subtype columns)
- `host_defense_diversity` (count of nonzero defense subtypes)
- `host_defense_abi_burden` (count of Abi-family subtypes)
- `tl17_rbp_family_count` (sum of RBP family presence columns)
- `host_capsule_abc_proxy_present` and `host_abc_serotype_proxy` (replaced by continuous capsule profile scores)

#### Infrastructure benchmarks

- **Figshare assemblies**: 403 FASTA files, 1.9GB zip, CC BY 4.0 license. Download: ~7 min via the "Download all" zip
  endpoint (`https://ndownloader.figshare.com/articles/25941691/versions/1`). Unzip: ~7 seconds. Downloaded on demand,
  stored in gitignored `lyzortx/data/assemblies/picard/`, skipped if already present.
- **CI image**: 9.5GB compressed (`full-bio` profile). The free GitHub Actions runner (14GB disk) cannot accommodate
  the assemblies baked into the image. DEPLOY tasks download at runtime instead.
- **Pharokka meta mode**: all 97 phage genomes annotated in 3 min 13s using `pharokka.py --meta --split` on a
  concatenated multi-FASTA, vs 1-2 hours with the current per-phage approach. The 40x speedup comes from running
  mmseqs2 indexing and profile search once instead of 97 times. This is a deferred optimization — the existing
  committed pharokka annotations (4.5MB) are small enough to keep.

#### Task structure and gating

7 tasks (DEPLOY01-07), reorganized from an initial 8-task plan based on review of acceptance criteria quality:

- **DEPLOY01** (download): assembly download script, no manifest
- **DEPLOY02** (defense, **gate**): re-derive defense features from raw assemblies; if DefenseFinder disagreement
  with panel annotations exceeds 3 systems/host on average, stop and investigate before proceeding
- **DEPLOY03** (surface): re-derive host surface features with continuous scores; depends on DEPLOY02 gate clearing
- **DEPLOY04** (typing): re-derive host typing; depends on DEPLOY03 (sequential to avoid merge conflicts on shared
  host feature code)
- **DEPLOY05** (phage RBP): switch to continuous mmseqs scores; independent, can run in parallel with DEPLOY02-04
- **DEPLOY06** (retrain): 3-way comparison (TL18 baseline vs parity-only vs parity+gradient), lock decision
- **DEPLOY07** (wire inference): make training and inference call the exact same functions, validate zero-delta parity

Each feature task (DEPLOY02-05) outputs a schema manifest (JSON) listing column names and dtypes. DEPLOY06 validates
the joint schema before assembling the feature matrix.

#### Expected outcomes

The primary outcome is deployment integrity: zero delta between training features and inference features for any host
whose assembly is available. The secondary outcome is potentially richer signal from continuous scores. DEPLOY06
explicitly separates these with a 3-way ablation (TL18 baseline vs parity-only vs parity+gradients) so we can measure
whether gradients help independently of the parity fix.

### 2026-04-02: DEPLOY01 assembly download implementation

#### Executive summary

Implemented `download_picard_assemblies()` in `lyzortx/pipeline/deployment_paired_features/download_picard_assemblies.py`.
The function derives the ST02 validation host set from `data/interactions/raw/raw_interactions.csv`, downloads the
figshare "Download all" archive to `.scratch/`, extracts it into `lyzortx/data/assemblies/picard/`, and fails loudly if
any of the 369 ST02 bacteria are missing their assembly file.

#### Interpretation

- The committed raw interactions table contains 369 unique bacteria, which matches the ST02 host count referenced in
  the plan.
- `data/genomics/bacteria/picard_collection.csv` contains 403 Picard collection hosts, so the download target and the
  ST02 validation set are different but overlapping cohorts.
- The download path is guarded by a complete-cache check: if `lyzortx/data/assemblies/picard/` already contains 403
  FASTA files, the function skips the network request and only validates coverage.

#### Tests

- Unique ST02 bacteria extraction from a semicolon-delimited raw interactions table.
- Zip extraction and validation on a small synthetic archive.
- Cache-skip behavior when the expected FASTA count is already present.
- Loud failure when a required ST02 assembly is absent.
