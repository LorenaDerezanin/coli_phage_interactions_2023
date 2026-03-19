### 2026-03-17: TB03 hard-to-lyse strains by host traits

#### What we implemented in TB03

1. Added one reproducible TB03 analysis script:
   `lyzortx/research_notes/ad_hoc_analysis_code/hard_to_lyse_host_traits.py`.
2. Configured the script to write generated outputs under
   `lyzortx/generated_outputs/hard_to_lyse_host_traits/`:
   - `hard_to_lyse_strain_summary.csv`
   - `host_trait_low_susceptibility_summary.csv`
   - `tb03_summary.json`
3. Used `<=3` lytic phages as the low-susceptibility threshold to stay aligned with the prior narrow-susceptibility
   slice from TB02.
4. Used derived `O-type:H-type` serotype labels for the main serotype analysis because `ABC_serotype` is missing for
   `251 / 402` strains (`62.4%`) and would mostly measure metadata completeness instead of biology.

#### TB03 output summary

- Interaction-matrix strains analyzed: `402`.
- Zero-lysis strains: `12 / 402` (`2.99%`).
- Low-susceptibility strains (`<=3` lytic phages): `36 / 401` resolved strains (`8.98%`), with `S1-84` remaining
  ambiguous because missing assays do not rule in or rule out the threshold.
- Zero-lysis strains:
  `B156`, `B253`, `DEC2a`, `E_albertiiCIP107988T`, `FN-B4`, `FN-B7`, `H1-002-0060-C-T`, `H1-007-0015-D-G`,
  `NILS22`, `NILS24`, `ROAR205`, `ROAR220`.
- Field-level stratification of lytic-phage counts was significant for all three requested host metadata fields:
  - Serotype (`O:H`): Kruskal-Wallis `p = 1.44e-03`
  - Phylogroup: Kruskal-Wallis `p = 1.24e-09`
  - ST: Kruskal-Wallis `p = 1.71e-05`
- No individual trait value met the current multiple-testing cutoff (`q <= 0.1`).
- Strongest nominal enrichments among testable groups (`n >= 4`) were:
  - Phylogroup `Clade V`: `3 / 7` low-susceptibility (`42.9%`), odds ratio `8.20`, `q = 0.218`
  - Phylogroup `E. fergusonii`: `2 / 5` low-susceptibility (`40.0%`), odds ratio `7.10`, `q = 0.264`
  - ST `58`: `5 / 17` low-susceptibility (`29.4%`), odds ratio `4.74`, `q = 0.296`
- Broad-susceptibility counterexample:
  - Phylogroup `B2`: only `6 / 126` low-susceptibility (`4.8%`), median `27` lytic phages, odds ratio `0.41`

#### TB03 interpretation

1. Hard-to-lyse behavior is not random noise in the matrix; it tracks host background strongly at the field level for
   serotype, phylogroup, and ST.
2. The clearest nominal concentration is in `Clade V` (`42.9%` low susceptibility versus an `8.98%` resolved-panel
   baseline), but none of the tested trait values yet survive multiple-testing correction.
3. `ST58` is still the strongest recurring ST signal in the currently testable set, but the ST landscape is fragmented
   across many small sequence types, so the category-level evidence remains weak after correction.
4. Serotype still matters at the field level, but the effect is diffuse across many rare `O:H` categories rather than a
   single dominant high-risk serotype. That means serotype is better treated as part of a multifeature host context than
   as a standalone rule.
5. Immediate next-step implication for modeling: host background features should keep explicit phylogroup and ST
   encodings, and mechanistic follow-up should prioritize the `Clade V` and `ST58` subsets first, with
   `E. fergusonii` as a secondary nominal candidate that still needs more support.

### 2026-03-17: TB04 rescuer phages for narrow-susceptibility strains

#### What we implemented in TB04

1. Added one reproducible TB04 analysis script:
   `lyzortx/research_notes/ad_hoc_analysis_code/rescuer_phages_for_narrow_susceptibility.py`.
2. Reused the TB03 low-susceptibility definition so the rescuer slice stays consistent with prior Track B work:
   resolved strains with `<=3` lytic phages and no missing assays.
3. Configured the script to write generated outputs under
   `lyzortx/generated_outputs/rescuer_phages_for_narrow_susceptibility/`:
   - `narrow_strain_rescuer_summary.csv`
   - `rescuer_phage_summary.csv`
   - `rescuer_phage_group_summary.csv`
   - `tb04_summary.json`
4. Used two operational rescuer modes:
   - `exclusive`: the phage is the only lytic hit for that narrow strain
   - `shared`: the phage is one of `2-3` lytic hits for that narrow strain

#### TB04 output summary

- Resolved narrow-susceptibility strains analyzed: `36`.
- Rescue-mode split: `9` exclusive-rescue strains, `15` shared-rescue strains, and `12` non-rescued narrow strains.
- Rescuer phages: `19 / 96` panel phages (`19.8%`) have at least one lytic hit in the resolved narrow-susceptibility
  slice.
- Top rescuer phages by resolved narrow-strain coverage:
  - `AL505_Ev3`: `5` narrow strains rescued (`13.9%` of the narrow slice), `1` exclusive and `4` shared
  - `NIC06_P2`: `4` narrow strains rescued, with the highest exclusive count (`3`)
  - `536_P9`: `4` narrow strains rescued, all shared
  - `DIJ07_P2`: `4` narrow strains rescued, all shared
  - `LF82_P8`: `4` narrow strains rescued, all shared
- The top five rescuer phages together cover `16 / 36` resolved narrow-susceptibility strains (`44.4%`).
- Exclusive rescue remains strongly Myoviridae-skewed:
  - `8 / 9` exclusive-rescue strains are rescued by Myoviridae phages
  - the only non-Myoviridae exclusive rescuer is podophage `AN24_P4`
- Morphotype/family concentration among rescuer phages:
  - Myoviridae: `17` rescuer phages, `41` narrow-strain rescue events, `8` exclusive rescues
  - Podoviridae: `2` rescuer phages, `4` narrow-strain rescue events, `1` exclusive rescue
  - Siphoviridae: `0` rescuer phages
  - Straboviridae: `11 / 11` panel phages are rescuer phages, contributing `25` narrow-strain rescue events and `7`
    exclusive rescues
- Highest narrow-hit concentration among materially active rescuer phages:
  - `AN24_P4`: `3 / 43` total lysed strains are narrow (`6.98%`)
  - `AL505_Ev3`: `5 / 160` (`3.13%`)
  - `NIC06_P2`: `4 / 170` (`2.35%`)

#### TB04 interpretation

1. Narrow-strain rescue is concentrated in a minority of the panel (`19 / 96` phages), and `12 / 36` resolved narrow
   strains are not rescued at all, so these hard cases are not being solved uniformly by broad host-range phages.
2. Myoviridae, especially `Straboviridae`, still dominate the rescue landscape and provide nearly all exclusive saves,
   which is consistent with the earlier single-lyser signal from the paper gist and TB02.
3. The podophage `AN24_P4` matters despite modest absolute coverage because it has the highest narrow-hit concentration
   and supplies the only non-Myoviridae exclusive rescue in the resolved slice. That makes it a targeted exception,
   not noise.
4. The top rescuers combine broad-rescue specialists (`AL505_Ev3`, `NIC06_P2`) with shared-support phages
   (`536_P9`, `DIJ07_P2`, `LF82_P8`), suggesting that narrow-susceptibility coverage is partly driven by a small
   backbone set plus a few strain-specific add-ons rather than one universally dominant rescuer.
5. Immediate modeling implication: phage feature work should keep morphotype/family signals, but it also needs enough
   phage-specific capacity to preserve exceptions like `AN24_P4` and the `NIC06_P2` exclusive-rescue pattern instead of
   collapsing everything into a broad Myoviridae prior.

### 2026-03-19: TB05 dilution-response patterns per phage and per bacterial subgroup

#### What we implemented in TB05

1. Added one reproducible TB05 analysis script:
   `lyzortx/research_notes/ad_hoc_analysis_code/dilution_response_patterns.py`.
2. Reused the Track A dilution-preserving outputs so the analysis runs on canonical pair labels plus the underlying
   pair-by-dilution observation summary instead of recomputing this structure ad hoc.
3. Configured the script to write generated outputs under
   `lyzortx/generated_outputs/dilution_response_patterns/`:
   - `pair_dilution_response_features.csv`
   - `per_phage_dilution_response_summary.csv`
   - `bacterial_subgroup_dilution_response_summary.csv`
   - `tb05_summary.json`
4. Used two operational dilution-response summaries for lytic pairs:
   - `high_potency`: best lysis dilution at `-2` or `-4`
   - `multi_dilution_support`: at least two dilution levels with any interpretable positive observation
5. Reported bacterial subgroup summaries across three host metadata views:
   - phylogroup
   - ST
   - derived `O:H` serotype

#### TB05 output summary

- Resolved pair labels analyzed: `35,266`.
- Lytic pairs analyzed: `9,720`.
- Best-lysis dilution distribution among lytic pairs:
  - `0`: `4,428` (`45.6%`)
  - `-1`: `2,097` (`21.6%`)
  - `-2`: `2,323` (`23.9%`)
  - `-4`: `872` (`9.0%`)
- High-potency lysis (`-2` or `-4`) accounts for `3,195 / 9,720` lytic pairs (`32.9%`).
- Multi-dilution support accounts for `5,249 / 9,720` lytic pairs (`54.0%`).
- Top high-potency phages with at least `40` lytic pairs:
  - `412_P2`: `57.4%` high-potency, `77.8%` multi-dilution support, `q = 5.76e-04`
  - `NRG_11B1`: `56.9%` high-potency, `71.5%` multi-dilution support, `q = 4.84e-08`
  - `536_P1`: `53.3%` high-potency, `64.1%` multi-dilution support, `q = 2.37e-08`
  - `55989_P2`: `53.2%` high-potency, `70.5%` multi-dilution support, `q = 2.36e-09`
  - `536_P6`: `50.6%` high-potency, `70.7%` multi-dilution support, `q = 1.32e-05`
- Lowest-potency phages with at least `40` lytic pairs:
  - `LM08_P1`: `2.8%` high-potency
  - `LM02_P1`: `2.9%`
  - `536_P11`: `3.6%`
  - `LM08_P2`: `3.8%`
  - `BDX03_P2`: `4.8%`
- The high-potency end is mostly Myoviridae with one strong podophage outlier (`412_P2`), while the lowest-potency end
  is dominated by Siphoviridae.
- Phylogroup-level dilution-response extremes:
  - `F`: `44.7%` high-potency, `66.4%` multi-dilution support, `q = 6.84e-07`
  - `B2`: `34.6%` high-potency, `56.2%` multi-dilution support, `q = 6.77e-03`
  - `Clade I`: only `15.9%` high-potency and `38.2%` multi-dilution support, `q = 1.29e-10`
  - `E. albertii`: only `11.6%` high-potency and `30.2%` multi-dilution support, `q = 6.77e-03`
- Highest-potency ST groups with at least `40` lytic pairs:
  - `624`: `57.5%` high-potency, `77.5%` multi-dilution support, `q = 1.08e-02`
  - `56`: `53.2%` high-potency, `66.0%` multi-dilution support, `q = 1.14e-03`
  - `80`: `52.3%` high-potency, `66.2%` multi-dilution support, `q = 9.08e-03`
  - `706`: `51.7%` high-potency, `69.0%` multi-dilution support, `q = 1.48e-02`
  - `79`: `45.9%` high-potency, `66.7%` multi-dilution support, `q = 8.46e-04`

#### TB05 interpretation

1. Dilution response is not a trivial restatement of binary host range. A substantial minority of lytic pairs stay
   positive at `-2` or `-4`, and these high-potency hits are concentrated in a specific subset of phages rather than
   spread uniformly across the panel.
2. The phage-level extremes are biologically structured. Most of the strongest dilution performers are Myoviridae, but
   podophage `412_P2` is a real exception with the highest high-potency rate in the tested slice, so potency should not
   be collapsed into a pure morphotype rule.
3. Host background modulates dilution robustness as well as simple susceptibility. Phylogroup `F` is enriched for
   stronger dilution-surviving lysis, while `Clade I` and `E. albertii` are depleted both for high-potency hits and for
   multi-dilution support, marking them as especially fragile-response host contexts.
4. This sharpens the TB03 picture. `B2` remains a comparatively favorable host background not just because it has more
   lytic hits overall, but because those hits are modestly more potent than the panel baseline; `Clade I` and
   `E. albertii` look difficult in both breadth and potency.
5. Immediate modeling implication: future label and modeling work should preserve dilution potency and multi-dilution
   support as first-class targets or auxiliary tasks. A binary `any_lysis` label would discard signal that clearly
   separates both phages and host subgroups.

