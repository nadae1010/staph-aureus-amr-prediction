# Staphylococcus aureus Antimicrobial Resistance Prediction

## Project overview

This project develops a machine-learning pipeline for predicting antimicrobial resistance (AMR) in *Staphylococcus aureus* using genomic sequences and laboratory-based phenotype data from the Bacterial and Viral Bioinformatics Resource Center (BV-BRC). It adapts the AURA framework (Serajian et al.), originally developed for *Mycobacterium tuberculosis*, to a new organism.

The project is being completed as part of a master's thesis under the supervision of Dr. Christina Boucher at the University of Florida.

## Research objective

The goal is to train and evaluate models that predict whether a *Staphylococcus aureus* genome is resistant or susceptible to individual antibiotics. The project also examines model generalizability, phenotype-label consistency, class imbalance, interpretability, and prevention of data leakage.

## Data-selection criteria

* Organism: *Staphylococcus aureus* only
* Source: BV-BRC
* More than 1,500 samples per selected antibiotic
* Resistant phenotype ratio of at least 30%
* Only laboratory-based phenotype evidence retained
* Data downloaded without filtering and cleaned afterward
* Duplicate genome IDs identified and handled; where a genome had several records for one antibiotic, the first qualifying record was kept when reshaping (the number of conflicting records is being counted)
* Untested genome–antibiotic combinations recorded as `None`

## Phenotype dataset

The cleaned phenotype matrix contains one row per genome:

1. `Genome ID`
2. `Genome Name`
3. `Strain Name`
4. One phenotype column per antibiotic (`Resistant`, `Susceptible`, or `None`)

4,944 unique genomes met the phenotype criteria. Of these, **4,610 genome assemblies were successfully retrieved**; 334 could not be recovered from BV-BRC and are listed in `data/missing_ids.txt`. All modeling uses the 4,610 available genomes.

Per-antibiotic counts on the modeled set:

| Antibiotic | Resistant | Susceptible | Tested | Resistance ratio |
|---|---|---|---|---|
| methicillin | 1,100 | 804 | 1,904 | 57.8% |
| cefoxitin | 1,229 | 730 | 1,959 | 62.7% |
| erythromycin | 1,359 | 1,728 | 3,087 | 44.0% |
| ciprofloxacin | 1,292 | 1,725 | 3,017 | 42.8% |
| clindamycin | 830 | 977 | 1,807 | 45.9% |
| oxacillin | 602 | 833 | 1,435 | 42.0% |

Oxacillin fell below the 1,500-tested threshold after the download losses; retained on advisor's decision as a minor decrease.

## Genome sequence data

Genome IDs from the cleaned phenotype matrix were used to request nucleotide FASTA files from BV-BRC, downloaded on HiPerGator using the BV-BRC Browser Tools. Genome files use the `.fna` extension; note that gerbil (inside KMX) rejects that extension, so `.fasta` symlinks were created. The symlinks used to build the matrix (`/blue/boucher/nelseifi/staph_fasta_links/`) point to the lab's shared BV-BRC copy at `/blue/boucher/share/BV-BRC/staph/`.

Large `.fna` files and the feature matrix remain on HiPerGator and are not committed here. This repository contains genome ID lists, processing and modeling code, documentation, summary statistics, and small tabular results.

## Feature matrix

Built with KMX (github.com/M-Serajian/KMX): **4,610 genomes × 11,351,471 k-mers**, k=31, min=5, max=3000, 90.6% sparse, stored as CSR on HiPerGator at `/blue/boucher/nelseifi/kmx_output/`. Chi-square ranking uses presence/absence rather than raw counts, so that multi-copy repetitive k-mers do not dominate the ranking.

The ranking score (`chi2_rank.py`) is a carrier-conditioned chi-square: for each k-mer it compares the number of resistant and susceptible genomes carrying it with the numbers expected from the overall resistant fraction, and ignores non-carriers. This is the form scikit-learn's `chi2` feature selector computes. It is **not** the statistic used in the AURA paper, whose code (`src/chi_square_test/chi_square_test.py` in the AURA repository) computes the full 2x2 chi-square with Yates' continuity correction. The two are compared directly in *The 2x2 chi-square alternative* below.

## Repository structure

```text
staph-aureus-amr-prediction/
├── README.md
├── data/
│   ├── README.md
│   ├── FINAL_staph_aureus_AMR_phenotypes.csv
│   ├── FILTERED_staph_aureus_AMR_phenotypes.csv
│   ├── FINAL_staph_aureus_genome_ids.txt
│   ├── phenotypes_matrix_aligned.csv
│   ├── AMR_summary_stats.csv
│   ├── Additional Antibiotic Eligibility Review-2.xlsx
│   ├── missing_ids.txt              # 334 genomes not recovered from BV-BRC
│   ├── genome_sources.csv           # genome -> source study mapping
│   ├── lineage_clusters.csv         # genome -> k-mer similarity cluster
│   ├── contigs_all.tsv              # assembled top-feature contigs
│   ├── long_contigs.fasta           # same, FASTA for BLAST
│   ├── external_accessions.txt      # 1,235 NCBI assembly accessions
│   ├── external_phenotypes.tsv      # NCBI AST phenotypes
│   ├── external_results.json        # external validation metrics
│   ├── dedup_report.tsv             # per-genome Jaccard vs training set
│   ├── loci_cefoxitin.tsv           # presence patterns, per antibiotic
│   ├── loci_ciprofloxacin.tsv
│   ├── loci_clindamycin.tsv
│   ├── loci_erythromycin.tsv
│   ├── loci_oxacillin.tsv
│   ├── compare_<antibiotic>.tsv     # carrier-conditioned vs 2x2 chi-square ranks
│   ├── seed_variance.tsv            # ten-seed external AUROC, all models
│   ├── stratified_clinda_results.txt
│   ├── erm_followup_results.txt
│   ├── amrfinder_summary_results.txt # AMRFinderPlus report
│   ├── amrfinder_calls.tsv          # AMRFinderPlus gene calls per genome
│   ├── amrfinder_elements.tsv       # every element called, with drug class
│   ├── amrfinder_contig_labels.txt  # AMRFinderPlus labels for the pipeline's contigs
│   ├── staph_results/               # random-split results, all feature counts
│   ├── staph_results_source/        # held-out-study results
│   ├── staph_results_source_cal/    # threshold-calibrated variant
│   ├── staph_results_models/        # five-model comparison
│   ├── staph_results_lineage/       # lineage-grouped results
│   └── staph_interpret/             # top-feature extraction output
└── src/
    ├── README.md
    ├── clean_amr.py                 # phenotype cleaning
    ├── make_one_final_csv.py        # wide phenotype matrix
    ├── make_amr_stats.py            # summary statistics
    ├── run_kmx.sh                   # k-mer matrix (GPU Slurm job)
    ├── make_splits.py               # strain-grouped 70/10/20 splits
    ├── chi2_rank.py                 # chi-square feature ranking
    ├── train_pilot.py               # XGBoost training and evaluation
    ├── model_compare.py             # five-model comparison
    ├── kmer_interpret.py            # top-feature extraction
    ├── kmer_contigs.py              # contig assembly from overlapping k-mers
    ├── mlsb_test.py                 # erm(C) vs mecA carriage vs MLSb phenotypes
    ├── source_holdout.py            # cross-study evaluation
    ├── source_holdout_cal.py        # threshold-calibrated variant
    ├── lineage_cluster.py           # k-mer similarity clustering
    ├── make_splits_lineage.py       # lineage-grouped 80/20 splits
    ├── train_lineage.py             # training on lineage-grouped splits
    ├── export_model.py              # final models on all labelled genomes
    ├── featurize_external.py        # k-mer presence over external assemblies
    ├── validate_external.py         # deduplication and external scoring
    ├── score_external.py            # standalone FASTA-manifest scorer
    ├── linkage_external.py          # locus-phenotype association, both datasets
    ├── count_loci.py                # presence patterns (SUPERSEDED, see below)
    ├── pattern_structure.py         # patterns ranked by chi-square, with structure
    ├── cipro_patterns.py            # ciprofloxacin pattern-length investigation
    ├── identify_top.py              # assemble and label the top-N feature set
    ├── retrain_without_meca.py      # partial mecA exclusion (SUPERSEDED)
    ├── retrain_v2.py                # full beta-lactam feature-set exclusion
    ├── bootstrap_variants.py        # paired bootstrap (evaluation variance only)
    ├── bootstrap_v2.py              # same, for the v2 variants
    ├── seed_variance.py             # fitting-variance noise floor
    ├── seed_variance_all.py         # ten-seed refits, all externally testable models
    ├── chi2_compare.py              # carrier-conditioned vs 2x2 chi-square ranking
    ├── stratified_clinda.py         # mecA-stratified clindamycin diagnostic
    ├── erm_followup.py              # erm(B)/erm(C) coverage and external odds ratios
    ├── run_amrfinder.sh             # AMRFinderPlus on all genomes (Slurm array)
    └── amrfinder_summary.py         # AMRFinderPlus gene calls vs phenotypes
```

## Data-processing workflow

1. Download the complete BV-BRC AMR phenotype table as CSV.
2. Count unique BV-BRC genome IDs.
3. Filter records to *Staphylococcus aureus*.
4. Retain laboratory-based phenotype evidence only.
5. Standardize phenotype labels.
6. Reshape into one row per genome, one column per antibiotic, keeping the first qualifying record where a genome has several.
7. Assign `None` where an antibiotic was not tested.
8. Calculate resistant, susceptible, and untested counts per antibiotic.
9. Select antibiotics meeting the sample-size and resistance-ratio criteria.
10. Download the corresponding `.fna` genome sequences on HiPerGator.
11. Build the k-mer feature matrix with KMX.
12. Align phenotype rows to matrix row order (`phenotypes_matrix_aligned.csv`).
13. Create training, validation, and test partitions without genome or strain leakage.
14. Rank features by carrier-conditioned chi-square on training rows only.
15. Train and evaluate antibiotic-specific prediction models.
16. Export final models trained on all labelled genomes and validate on an
    independent NCBI collection.

## Data-leakage prevention

Training, validation, and test sets are separated at the genome level; records representing the same genome never appear in multiple partitions. Splits are grouped by strain name so that repeated strain records stay within one partition, and each split is verified by assertion before use. All preprocessing — including chi-square feature selection — is fitted on training data only and applied unchanged to validation and test sets.

Partitions are 70/10/20, repeated over five fixed seeds (0–4), with class counts recorded for every split. A second, stricter split design groups genomes by k-mer similarity cluster (see *Lineage-grouped evaluation*). A third check applies to the external collection: because BV-BRC ingests public assemblies, every external genome is compared to every training genome by Jaccard similarity over a 20,000 k-mer panel and near-identical genomes are excluded before scoring (see *External validation*).

## Model evaluation

Models are assessed with accuracy, precision, recall, F1, ROC-AUC, precision–recall AUC, confusion matrices, and class-specific performance, reported as mean ± standard deviation across the five seeds.

**Single-fit comparisons at large feature counts are not reliable in this
pipeline.** Clindamycin's external AUROC varies from 0.758 to 0.804 across ten
random seeds with no other change (mean 0.784, sd 0.014). Any comparison
between two models must therefore be made between seed distributions, not
between point estimates. See *Fitting variance* below.

## Results

### Predictive performance

XGBoost, top 10,000 k-mers, mean over 5 seeds:

| Antibiotic | F1 | AUROC | AUPRC |
|---|---|---|---|
| methicillin | 0.994 | 0.997 | 0.998 |
| cefoxitin | 0.956 | 0.989 | 0.994 |
| ciprofloxacin | 0.944 | 0.971 | 0.961 |
| oxacillin | 0.895 | 0.948 | 0.940 |
| erythromycin | 0.888 | 0.931 | 0.910 |
| clindamycin | 0.824 | 0.892 | 0.876 |

### Model comparison

Across all six antibiotics, LASSO logistic regression, linear SVM, random forest, and XGBoost perform within ~1 point of one another; naive Bayes is substantially worse on ciprofloxacin (F1 0.578 vs 0.945). LASSO matches XGBoost while training ~10× faster and offering directly interpretable coefficients.

### Mechanism recovery

Stable top-ranked k-mers (top 5,000 in at least four of five seeds) were
assembled into contigs on 30 bp overlap (`kmer_contigs.py`), then identified by
translation and by BLAST. No alignment, reference genome, or gene annotation is
used anywhere in the prediction pipeline. A reference database (AMRFinderPlus)
was used only afterwards, to check these identifications and explain the
clindamycin failure (see *Reference-based annotation*).

| Antibiotic | Top locus | Genetic context | R / S |
|---|---|---|---|
| methicillin | *mecA* (PBP2a) | SCCmec | 98.5% / 0.5% |
| oxacillin | *mecA* | SCCmec | 90.4% / 7.3% |
| cefoxitin | *mecA* | SCCmec | 88.0% / 12.9% |
| ciprofloxacin | *gyrA* S84L, *grlA* S80F | chromosomal point mutations | 86.1% / 3.4% |
| erythromycin | *erm*(C) | 2,473 bp plasmid | 48.1% / 2.4% |
| clindamycin | *mecA*, N-terminal segment | SCCmec | 68.7% / 12.8% |

**Contig length distinguishes mechanism class.** Ciprofloxacin's longest contig
is 75 bp while the others reach 2,500–2,800 bp. A point mutation alters exactly
*k* k-mers, which assemble into a 2*k*−1 bp window; an acquired gene contributes
k-mers across its full length. Contig length therefore diagnoses point-mutation
versus acquired-gene resistance without annotation.

**Both alleles of the quinolone-resistance mutations were recovered.** The four
top ciprofloxacin contigs form two pairs differing at a single nucleotide:
*gyrA* S84L (86.1% R / 3.4% S) with wild-type Ser84 (7.8% R / 95.5% S), and
*grlA* S80F (81.4% R / 4.7% S) with wild-type Ser80 (4.3% R / 91.0% S). Each is
61 bp from exactly 31 k-mers, the arithmetic signature of a SNP. Ranked by
chi-square on all labelled genomes these occupy ranks 0 and 7 of 11,351,471.

**Erythromycin's determinant is plasmid-borne.** Its top contig closes on
itself with a 30 bp overlap, giving a 2,473 bp circular element carrying a
244-aa ORF with the GSGKG (SAM-binding) and NIPY (N6-adenine methyltransferase)
motifs, confirmed by blastx as Erm(C) at 100% identity.

**mecA assembles as two non-overlapping contigs.** The assembly yields a
2,829 bp contig and a 628–742 bp contig that share no k-mers. blastx on the
larger returns PBP2a residues 228–649 at 100% identity; the smaller contains
the mecA ribosome binding site and the first ~227 bp of coding sequence
(blastn 100% identity to the mecR1/mecA region of PV938955.1, positions
108–365). The gap spans roughly residues 76–227, where those k-mers did not
reach the top features. Both segments are carried by essentially the same
genomes (2,958 and 2,959 of 4,610 at a half-the-k-mers threshold), so the
N-terminal region is more allelically variable rather than more often deleted:
exact 31-mers fail where sequence diverges, which lowers its carriage among
both resistant and susceptible isolates.

**Clindamycin's top feature is *mecA*, not an MLSb determinant.** Ranked by
chi-square on all labelled genomes, its strongest feature is the N-terminal
mecA segment (chi2 364.3, rank 0, 68.7% of resistant and 12.8% of susceptible),
with the C-terminal segment following at rank 39 (chi2 350.0, 82.2% / 20.5%);
*erm*(C) does not appear among its leading features. The cause is coverage
under the feature-selection chi-square, which conditions on feature-positive
samples: *mecA* is carried by 82.3% of clindamycin-resistant isolates (21.6% of
susceptible), while *erm*(C) reaches only 29.5% (7.9%). The N-terminal segment
outranks the C-terminal one despite lower coverage because it is cleaner on the
susceptible side, which the carrier-conditioned statistic rewards. Among
*mecA*-negative isolates 708 of 912 are susceptible to both MLSb drugs, so
*mecA* marks a broadly co-resistant MRSA population rather than conferring
clindamycin resistance. The specific link was later identified as *erm*(A),
which travels with *mecA* in the training genomes (see *Reference-based
annotation*). Inducible expression contributes modestly — 40 of 285
*erm*(C) carriers (14%) are erythromycin-resistant but clindamycin-susceptible,
which degrades *erm*(C)'s clindamycin specificity but is not the dominant cause
(`mlsb_test.py`).

**Erythromycin's other leading contigs are aminoglycoside resistance genes.**
Contigs 2 (338 bp) and 5 (523 bp) are ANT(6)-Ia and contig 3 (1,053 bp) is
APH(3')-IIIa, each at 100% identity by blastx. Both confer aminoglycoside, not
macrolide, resistance; they are carried by 29-34% of erythromycin-resistant
and 1-2% of susceptible genomes because they travel with multidrug-resistant
MRSA: AMRFinderPlus finds *mecA* in 95% of *aph(3')-IIIa* carriers, which also
carry *ant(6)-Ia* (98%) and *sat4* (74%). They are linked markers within a model that
nonetheless transfers externally, because its top-ranked feature is the causal
erm(C).

### Feature-count requirements track resistance architecture

AUROC as a function of the number of chi-square-selected k-mers:

| Antibiotic | 1k | 5k | 10k | 50k | 200k | 500k |
|---|---|---|---|---|---|---|
| methicillin | 0.995 | 0.996 | 0.997 | 0.999 | — | — |
| cefoxitin | 0.981 | 0.987 | 0.989 | 0.989 | — | — |
| ciprofloxacin | 0.967 | 0.969 | 0.970 | 0.972 | — | — |
| oxacillin | 0.916 | 0.933 | 0.948 | 0.954 | — | — |
| erythromycin | 0.730 | 0.922 | 0.931 | 0.964 | — | — |
| clindamycin | 0.826 | 0.887 | 0.892 | 0.919 | 0.942 | 0.944 |

Single-determinant mechanisms saturate by ~1,000 features. The MLSb antibiotics (erythromycin, clindamycin) require roughly 50× more, consistent with resistance distributed across multiple *erm* genes rather than one determinant. Clindamycin plateaus near 200,000. Clindamycin's instability under refitting is related but not reducible to
feature count: see *Fitting variance*.

### Lineage-grouped evaluation

Strain-name grouping affects only 19 genomes, so splits were rebuilt using k-mer similarity clusters as a lineage proxy (394 clusters at Jaccard distance 0.10; `lineage_cluster.py`). Under lineage grouping, near-identical isolates cannot span train and test.

AUROC, strain-grouped (70/10/20, early stopping) vs lineage-grouped (80/20, fixed trees):

| Antibiotic | Strain @10k | Lineage @10k | Lineage @50k | Lineage @200k |
|---|---|---|---|---|
| cefoxitin | 0.989 | 0.985 | — | — |
| methicillin* | 0.997 | 0.981 | — | — |
| ciprofloxacin | 0.971 | 0.949 | — | — |
| oxacillin | 0.948 | 0.883 | — | — |
| erythromycin | 0.931 | 0.907 | 0.954 | 0.963 |
| clindamycin | 0.892 | 0.755 | 0.857 | 0.851 |

\* methicillin lineage splits are degenerate: one 460-genome cluster forces test sets that are 87–95% resistant in every seed, so its five seeds are not independent replicates.

Two effects separate out. First, an insufficient feature budget can masquerade as lineage sensitivity: at 10,000 features the MLSb antibiotics appear to depend on population structure, but erythromycin fully recovers at 50,000 (0.954) and exceeds its strain-grouped score at 200,000 (0.963), with seed variance falling from ±0.054 to ±0.014. Second, clindamycin retains a residual lineage effect after the feature budget is corrected (0.857 at 50k, plateauing, versus 0.892 strain-grouped, seed sd ±0.071). Because *mecA* carriage is lineage-associated and *mecA* is clindamycin's top-ranked feature, this residual is the within-dataset shadow of the linkage dependence that the external validation makes explicit.

Practical consequence: feature counts should be set per antibiotic, with ~50,000 as a floor for MLSb drugs rather than a single global value.

### Cross-study generalization

Phenotypes derive from ~15 source studies with sharply differing resistance prevalence; source study alone predicts the label at 72–85% accuracy. Held-out-study evaluation shows discrimination largely survives (AUROC 0.82–0.99), but decision thresholds do not transfer. Calibrating thresholds on other training studies did not help and degraded F1 for four of six antibiotics (`source_holdout_cal.py`, documented negative result). AUROC and AUPRC are the appropriate cross-study metrics.

### External validation

Final models were retrained on all labelled genomes (`export_model.py`) with
per-antibiotic feature budgets, then applied to **1,235 *S. aureus* assemblies
from the NCBI Pathogen Detection AST Browser**, a collection independent of
BV-BRC. Because BV-BRC ingests public assemblies, each external genome was
compared to all 4,610 training genomes by Jaccard similarity over a 20,000
k-mer panel; **87 genomes at or above 0.99 were excluded**, leaving 1,148.
Results below are post-deduplication.

| Antibiotic | n | ext % R | AUROC (10 seeds) | sd | Brier | Brier skill |
|---|---|---|---|---|---|---|
| ciprofloxacin | 729 | 4.9% | **0.9912** | 0.0008 | 0.0140 | 70% |
| oxacillin | 1,132 | 29.1% | **0.9875** | 0.0006 | 0.0339 | 84% |
| cefoxitin | 337 | 33.2% | **0.9647** | 0.0025 | 0.0816 | 63% |
| erythromycin | 1,118 | 41.0% | **0.9606** | 0.0009 | 0.0629 | 74% |
| clindamycin | 1,109 | 30.0% | **0.7841** | 0.0138 | 0.1587 | 24% |

Each figure is the mean of ten refits differing only in random seed
(`seed_variance_all.py`). Brier skill is measured against predicting the
external base rate for every genome.

Four of five models transfer at or near their internal performance; ciprofloxacin
and oxacillin exceed it. **Methicillin could not be validated externally**: NCBI
carries no methicillin AST records, which reflects the same clinical reality —
methicillin has been replaced by oxacillin and cefoxitin in routine testing —
that made its BV-BRC labels thin.

The results reproduced exactly on a second full re-featurization against a
different k-mer union, to four decimal places. That reproducibility is of the
scoring pipeline given a fixed fitted model; it does not extend to refitting.

### Linkage structure explains the clindamycin failure

Clindamycin is the only model that fails externally, and its top features are
*mecA* rather than an MLSb determinant. Linkage is a property of a population,
not of the organism, so a collection with different MRSA composition should
break it. Carriage of each locus was scored identically in both datasets (at
least half the locus's k-mers present) and cross-tabulated against phenotype
(`linkage_external.py`):

| Locus – phenotype | Training OR | External OR |
|---|---|---|
| *mecA* – cefoxitin | 118 | 4,964 |
| *mecA* – oxacillin | 112 | 4,137 |
| *mecA* – erythromycin | 10.8 | 3.0 |
| *mecA* – clindamycin | 16.8 | **1.5** |
| *erm*(C) – erythromycin | 29.2 | 277 |
| *erm*(C) – clindamycin | 4.9 | 162 |

Causal determinants strengthen in the external population; the linkage marker
collapses. Externally *erm*(C) is nearly perfectly specific for clindamycin —
one carrier among 833 susceptible isolates — but covers only 22.5% of resistant
isolates, which is the direct cause of the 0.456 external recall.

### Feature strength predicts external transfer

Ranked by chi-square on all labelled genomes (`pattern_structure.py`), each
model's single strongest feature orders the models exactly as external AUROC
does (ten-seed means):

| Antibiotic | best feature | chi2 | R carriage | S carriage | OR | external AUROC |
|---|---|---|---|---|---|---|
| ciprofloxacin | *gyrA* S84L | 1303.0 | 86.1% | 3.4% | 175.7 | 0.991 |
| erythromycin | *erm*(C) | 703.2 | 48.1% | 2.4% | 36.7 | 0.961 |
| oxacillin | *mecA* | 571.6 | 90.4% | 7.3% | 116.9 | 0.988 |
| cefoxitin | *mecA* | 446.7 | 94.7% | 15.1% | 99.8 | 0.965 |
| clindamycin | *mecA* N-term | 364.3 | 68.7% | 12.8% | 14.9 | 0.784 |

Sorted by odds ratio — 175.7, 116.9, 99.8, 36.7, 14.9 — the ordering matches
external AUROC exactly (0.991, 0.988, 0.965, 0.961, 0.784). With five points
and the relationship identified after the fact, a perfect ordering arises by
chance with probability about 0.008; this is suggestive, not established, and
would need more antibiotics to test.

### Locus-level feature structure

Collapsing each model's top 10,000 features into distinct presence patterns
shows how many independent bets each model carries. Exact ties in chi-square
indicate k-mers with identical presence across all genomes.

| Antibiotic | distinct patterns | external AUROC (ten-seed mean) |
|---|---|---|
| cefoxitin | 959 | 0.965 |
| erythromycin | 742 | 0.961 |
| ciprofloxacin | 563 | 0.991 |
| oxacillin | 483 | 0.988 |
| clindamycin | 325 | 0.784 |

Concentration alone does not predict transfer: oxacillin has the second-fewest
patterns and the second-best external AUROC.

**A shared presence vector does not imply a contiguous locus.** Assembling each
pattern (`pattern_structure.py`) separates two cases. For mecA and *erm*(C) a
pattern's k-mers join into one long contig — an acquired gene, present or
absent as a unit. For ciprofloxacin they do not: its largest pattern holds 933
k-mers that assemble into **42 separate 61 bp contigs**, unlinked point
mutations co-occurring because they are inherited together in a clonal lineage.
Those lineage haplotypes are real but rank 2,434 and below, far beneath the
causal *gyrA*/*grlA* mutations at ranks 0 and 7.

An earlier version of this analysis (`count_loci.py`) ranked patterns by size
rather than chi-square and reported a locus length of *n* k-mers + *k* − 1,
which assumes contiguity. Both were wrong: size is not predictiveness, and the
implied lengths were arithmetic artifacts. `pattern_structure.py` supersedes it
and `count_loci.py` is retained only for provenance.

### Fitting variance is confined to the failing model

Every model was refit ten times with nothing changed but the random seed and
rescored on the same external genomes.

| Model | features | best-feature OR | AUROC sd | range |
|---|---|---|---|---|
| oxacillin | 50,000 | 116.9 | 0.0006 | 0.0023 |
| ciprofloxacin | 5,000 | 175.7 | 0.0008 | 0.0022 |
| erythromycin | 50,000 | 36.7 | 0.0009 | 0.0027 |
| cefoxitin | 5,000 | 99.8 | 0.0025 | 0.0081 |
| clindamycin | 200,000 | 14.9 | 0.0138 | 0.0457 |

The four models that transfer are reproducible to three decimal places.
Clindamycin's variance is 15 to 23 times larger.

Feature count alone does not explain this: oxacillin uses 50,000 features and
has the lowest variance of the five, while cefoxitin uses 5,000 and has four
times as much. What distinguishes clindamycin is the weakness of its strongest
feature (OR 14.9 against 36.7-175.7 elsewhere). Where no feature is decisive,
many are interchangeable and a bagged model wanders between them. Cefoxitin's
slightly elevated variance reflects its smaller external set (n=337).

The practical consequence stands: at clindamycin's feature count, single-fit
comparisons cannot resolve interventions smaller than about 0.03 AUROC, which
is why the exclusion experiments are reported as nulls. For the other four
antibiotics, single fits would have been adequate -- but that could not be
known without measuring it.


### Excluding the confounded feature: a negative result

Because clindamycin's leading features are *mecA*, the natural test is whether
removing them improves external transfer. Two attempts were made.

The first (`retrain_without_meca.py`) excluded the 2,799 k-mers of the assembled
*mecA* block, plus a |φ| > 0.7 variant. This exclusion was **incomplete**: mecA
assembles as two non-overlapping contigs and only the C-terminal one was
removed, leaving the N-terminal segment — clindamycin's rank-0 feature — in
place. The diagnostic was visible at the time and missed: the top chi-square
was 364.3 before exclusion and exactly 364.3 after.

The second (`retrain_v2.py`) replaced locus naming with a rule statable in
advance: exclude every feature used by the beta-lactam models (top 50,000 each
from methicillin, oxacillin and cefoxitin; 119,788 k-mers). Those predict a
different drug class, so shared signal is linkage by construction. This cut
did bite — clindamycin's top chi-square fell 364.3 → 307.4 — while the
erythromycin control was untouched at 703.2.

Against the baseline's ten-seed mean of 0.7841 ± 0.0138:

| Model | ten-seed mean | sd | vs baseline |
|---|---|---|---|
| clindamycin baseline | 0.7841 | 0.0138 | — |
| beta-lactam set excluded (loci) | 0.7812 | 0.0137 | −0.0028 (0.2 sd) |
| plus phi > 0.5 (phi) | 0.7737 | 0.0105 | −0.0103 (0.8 sd) |

**Both excluded models fall below the unexcluded baseline's expected value.**
The apparent improvements reported earlier (+0.0168 and +0.0242, each with a
bootstrap interval excluding zero) were artifacts of the baseline having been
fit at seed 0, which is the minimum of its own ten-seed distribution.

The conclusion is therefore negative: **excluding the confounded feature set
does not improve external generalization.** Whether a real effect exists
beneath a ±0.014 noise floor is a question this design cannot answer. The
confounding itself is independently established by the linkage odds ratios
above and does not rest on these experiments.

The methodological point generalizes beyond this organism: at feature counts
where chi-square scores are nearly flat — clindamycin's rank 1 scores 270.3 and
its rank 1,000 scores 252.0 — many features are interchangeable, bagged trees
wander between them, and single-fit comparisons cannot resolve interventions of
this size.

### The 2x2 chi-square alternative: a pre-registered negative result

The carrier-conditioned chi-square ranks features by coverage, which is why it
selects mecA over erm(C) for clindamycin. The obvious remedy is the standard
2x2 chi-square, which counts non-carriers and so penalises false positives.
The 2x2 form is also the statistic the AURA paper uses (with Yates' continuity
correction, omitted here), so arm B is effectively AURA's own criterion.
This was tested with the decision rule fixed in advance: arm B would be judged
an improvement only if its ten-seed mean exceeded arm A's by more than two
pooled standard deviations.

The premise failed at the first diagnostic (`chi2_compare.py`), so no models
were retrained.

| clindamycin locus | rank, carrier-conditioned | rank, standard 2x2 |
|---|---|---|
| mecA (C-terminal) | 45 | **6** |
| mecA (N-terminal) | 2 | 129 |
| *erm*(C) | 33,112 | **37,732** |

erm(C) ranks *lower* under 2x2, not higher. The reason is false negatives,
which the carrier-conditioned form ignores and the 2x2 does not: mecA is absent
from only 147 of 830 clindamycin-resistant isolates, while erm(C) is absent
from 524. Phi against phenotype is 0.61 for mecA and 0.35 for erm(C).

**Within the training population, mecA is genuinely the better predictor of
clindamycin resistance.** Two criteria with opposing biases both select it,
because it is correctly the better choice there.

The alternative also degrades two models that currently work. For erythromycin,
erm(C) falls from rank 0 to rank 57 and mecA occupies the top five. For
ciprofloxacin, mecA — a confounder there too, carried by 93.7% of resistant
isolates — rises from rank 327,370 to rank 161, while the causal gyrA/grlA
mutation falls from rank 0 to 16. The carrier-conditioned criterion suppresses
that confounder; the 2x2 does not.

The conclusion generalises beyond feature selection: **confounding by population
structure cannot be detected or corrected from within the training population.**
Any criterion optimising within-population prediction selects the confounder,
because the confounder predicts better there. Only external validation exposes
it. Retaining the carrier-conditioned chi-square, rather than switching to the
2x2 form AURA uses, is therefore an evidence-backed choice. (Earlier versions
of this README called the carrier-conditioned form "AURA's"; the AURA paper and
code use the 2x2 test with Yates' correction.)

That ciprofloxacin result is independently informative: mecA is a confounder for ciprofloxacin as well, carried by 93.7% of resistant isolates through the same hospital-lineage association. The carrier-conditioned criterion buries it at rank 327,370 while ranking the causal mutation first. Confounding is therefore present in models that generalise successfully; clindamycin is distinguished not by having a confounder but by the confounder outranking the determinant.

**Confounders are present in models that transfer.** Across three
antibiotics, the deciding factor is not whether a linked marker exists but
whether the causal determinant outranks it:

| Model | Linked marker in feature set | Top-ranked feature | External AUROC |
|---|---|---|---|
| ciprofloxacin | mecA (rank 327,370) | gyrA S84L, causal (rank 0) | 0.991 |
| erythromycin | ANT(6)-Ia, APH(3')-IIIa | erm(C), causal (rank 0) | 0.961 |
| clindamycin | mecA (rank 2) | mecA, linked | 0.784 |

### Stratified clindamycin diagnostic: a pre-registered negative result

Both previous fixes failed for the same reason: within the training population
mecA genuinely predicts clindamycin better than any causal determinant, so any
criterion optimising pooled prediction selects it. The standard response to a
known confounder is to stratify on it. Within mecA-negative genomes mecA cannot
contribute, so any remaining signal must come from something else.

The decision rule was fixed in advance: proceed to stratified feature selection
only if the mecA-negative stratum reached a mean cross-validated AUROC of at
least 0.80, matching the current external mean. Cross-validation was 5-fold,
grouped by lineage cluster, over 3 seeds, with feature selection refit inside
every fold (`stratified_clinda.py`).

| Stratum | n | R | S | AUROC, 5,000 features | AUROC, 50,000 features |
|---|---|---|---|---|---|
| mecA-negative | 913 | 147 | 766 | 0.621 ± 0.112 | **0.775 ± 0.059** |
| mecA-positive | 894 | 683 | 211 | uninterpretable* | uninterpretable* |

**Verdict: FAIL** (best 0.775 against a bar of 0.80). Without mecA there is real
clindamycin signal in methicillin-susceptible genomes, well above chance, but
not enough to justify building stratified selection into the pipeline.

\* A single 460-genome lineage cluster dominates the mecA-positive stratum, so
lineage-grouped folds were degenerate: one fold contained only one class and
was skipped, and the remainder ranged from 0.00 to 0.97.

**Stratification did surface a determinant the pooled analysis missed.** The top
mecA-negative contig (523 bp; 28.6% of resistant, 2.0% of susceptible) was
identified by blastx as Erm(B) at 100% identity. erm(B) methylates the ribosome
and confers resistance to both macrolides and lincosamides. In the pooled
ranking it was buried beneath mecA; it became visible only once mecA could not
contribute. erm(C) rose from rank 33,112 to 18,456 in the same stratum.

Inside the mecA-positive stratum, the leading features were markers of
*susceptibility* (about 47% of susceptible and 3.5% of resistant isolates),
assembling to short contigs consistent with lineage SNPs rather than a gene:
within MRSA, the best clindamycin predictor is again a population marker.

### erm(B), and how much clindamycin resistance remains unexplained

Stratification surfaced erm(B) as a clindamycin determinant the pooled analysis
had missed. `erm_followup.py` tests it the way erm(C) and mecA were tested:
by whether its association with clindamycin survives a change of population.
Carriage is scored on 24 evenly spaced k-mers per locus (a genome carries the
locus if at least half are present), identically in both collections. External
genomes were read directly from the raw assemblies, because erm(B) was never
selected by any model and so is absent from the external feature file.

**erm(B) behaves like a causal determinant.** The prediction was stated before
the run: a causal gene keeps an external odds ratio whose 95% interval excludes
1, while a linked marker drifts toward 1.

| Locus | Training OR | External OR (deduplicated) | External carriers, R / S |
|---|---|---|---|
| *erm*(B) | 8.9 [6.0, 13.3] | **40.6 [2.3, 704.7]** | 8 / 0 |
| *erm*(C) | 4.9 [3.7, 6.4] | **461.4 [28.5, 7470.3]** | 76 / 0 |
| *mecA* | 16.8 [13.3, 21.2] | **1.5 [1.2, 2.0]** | 122 / 213 |

Both erm genes strengthen externally and neither appears in a single
susceptible NCBI genome; mecA collapses. The prediction holds for erm(B), but
the evidence is thin: it rests on 8 external carriers, which is why the
interval spans two orders of magnitude.

**The pooled analysis excluded erm(B) entirely.** Its best k-mer ranked
219,559th in the pooled clindamycin ranking, beyond the 200,000-feature budget
of the exported model, so the model never saw it. This is despite erm(B) being
the *more* specific of the two erm genes per carrier (OR 8.9 against erm(C)'s
4.9): the ranking rewarded erm(C)'s greater coverage, and mecA's greater still.

**erm(B) and erm(C) are alternatives, not companions.** Only 3 of 830
resistant training genomes carry both.

| Resistant genomes | erm(B) only | erm(C) only | both | neither | covered |
|---|---|---|---|---|---|
| training, all (830) | 177 | 242 | 3 | 408 | 50.8% |
| training, mecA-negative (147) | 41 | 36 | 1 | 69 | 53.1% |
| training, mecA-positive (683) | 136 | 206 | 2 | 339 | 50.4% |
| external, all (333) | 8 | 76 | 0 | 249 | **25.2%** |
| external, mecA-negative (211) | 3 | 28 | 0 | 180 | **14.7%** |

**The external resistant population is mostly mecA-negative.** 82.3% of
clindamycin-resistant training genomes carry mecA, against 36.6% of external
ones. In BV-BRC, clindamycin resistance largely sits in MRSA, so mecA works as
a proxy; in NCBI most of it sits in MSSA, so the proxy fails. This is the most
direct single explanation of the external collapse.

**Most external clindamycin resistance is unexplained by either erm gene.**
Three quarters of resistant external genomes carry neither. That resistance is
almost always to erythromycin as well: only 12 of 1,801 training genomes and 2
of 1,090 external genomes are clindamycin-resistant but erythromycin-
susceptible. Lincosamide-specific mechanisms such as lnu(A) or vga(A) therefore
account for very little. The pattern points to a third macrolide-lincosamide
determinant. Notably, the erythromycin model still transfers at 0.961 although
erm(B) and erm(C) together cover only about 19% of external
erythromycin-resistant genomes, so it must capture something that tracks this
third determinant. Reference-based annotation (next section) identifies it as
erm(A), plus erm(T) in the external collection.

**Inducible resistance is visible in the labels.** 134 training genomes are
erythromycin-resistant but clindamycin-susceptible, and 52 of them carry erm(B)
or erm(C). This is the inducible MLSb pattern: the gene is present, but
clindamycin resistance is only expressed, and only labelled, when induction is
tested for.

### Reference-based annotation: erm(A) and erm(T) fill the gap

To identify the missing determinant, NCBI AMRFinderPlus 4.2.7 (database
2026-10-29.1, `--organism Staphylococcus_aureus --plus`) was run on all 4,610
training and 1,235 external genomes (`run_amrfinder.sh`,
`amrfinder_summary.py`). This is the only step in the project that uses a
reference database, and it checks and explains the pipeline's results rather
than supplying features.

**The gene calls reproduce the k-mer findings.** mecA is called in 96.2% of
cefoxitin-resistant and 17.4% of susceptible training genomes (k-mer carriage:
94.7% / 15.1%), and mecA's clindamycin odds ratio falls from 16.7 in training
to 1.5 externally (k-mer: 16.8 to 1.5). Labelling the pipeline's own contigs
(`amrfinder_contig_labels.txt`) confirms both mecA segments, erm(C), erm(B)
and APH(3')-IIIa at 100% identity.

**Known genes explain about 92% of clindamycin resistance in both
collections.**

| Clindamycin-resistant genomes | Training (830) | External, deduplicated (333) |
|---|---|---|
| carry erm(B) or erm(C) | 482 (58.1%) | 86 (25.8%) |
| carry any known lincosamide gene | 765 (92.2%) | 309 (92.8%) |
| gap filled by | erm(A): 275 of 348 | erm(A) 139 and erm(T) 84, of 247 |
| still unexplained | 65 (7.8%); 60 also erythromycin-resistant | 24 (7.2%); 23 also erythromycin-resistant |

The erm(B)/erm(C) coverage agrees with the k-mer estimate above (50.8% /
25.2%); gene calls and k-mer carriage are different detectors, so small
differences are expected.

**erm(A) explains why mecA looked predictive.** erm(A) is strongly associated
with clindamycin resistance (OR 19.6 in training, 197 externally) and
co-occurs with spc, *ant(9)-Ia* (Jaccard 0.99): the gene content of transposon
Tn554. In training, 84.6% of erm(A) carriers are mecA-positive, and erm(A)
also travels with mecI (Jaccard 0.64), consistent with Tn554 inside the older
hospital SCCmec types, as in the SCCmec type II reference strain N315 (Ito et
al., PMC89295). Externally only 43.6% of erm(A) carriers carry mecA. The
physical placement is inferred from co-occurrence; it was not mapped in these
assemblies.

**erm(T) cannot be learned from BV-BRC.** erm(T) is carried by 25.2% of
resistant and 0.3% of susceptible external genomes, but by no
clindamycin-resistant training genome. All its external carriers are
mecA-negative and co-occur with *glpT* F3I (Jaccard 0.91), i.e. a single
lineage. erm(T) has been proposed as a marker of human-associated CC398 MSSA
(Mama et al. 2021); MLST has not been run to confirm this here.

**Even curated genes select mecA within the training population.** Ranking
AMRFinderPlus elements against clindamycin by 2x2 chi-square puts mecA first
in training (chi2 659) and erm(A) sixth (337), behind lineage markers such as
*parC* S80F and *ant(6)-Ia*. Externally erm(A), its partner spc, erm(T) and
erm(C) lead. This reproduces, with annotated genes, the conclusion above:
confounding by population structure cannot be corrected from inside the
training population.

**The pipeline did carry an erm(A) signal.** Labelled at a 10% coverage
threshold, clindamycin contig 11 (118 bp) is an erm(A) fragment, listed after
both mecA segments and the aminoglycoside markers *ant(6)-Ia* (contig 10) and
*sat4* (contig 9). It was too short to identify earlier.

**Erythromycin.** Known macrolide genes explain 92.5% of training and 94.3% of
external erythromycin resistance (excluding *lmrS*, a chromosomal efflux pump
carried by 94% of genomes regardless of phenotype).

**Label quality.** 17.4% of clindamycin-susceptible training genomes carry a
lincosamide gene, against 1.8% externally, while only 4.6% of
erythromycin-susceptible training genomes carry a macrolide gene. This fits
inducible clindamycin resistance that was not D-tested in BV-BRC, but does
not prove it.

## Dataset versions

BV-BRC has no release numbers, so download and file dates stand in for
versions.

| Item | Source | Date or version |
|---|---|---|
| Genome ID list | BV-BRC (`FINAL_staph_aureus_genome_ids.txt`) | 2026-05-04 |
| Phenotype tables (`FINAL_`, `FILTERED_`) | BV-BRC | written 2026-09-03; original export date to be confirmed |
| Training genomes | Lab's shared BV-BRC copy, `/blue/boucher/share/BV-BRC/staph/` | files dated 2026-08-28 |
| External genomes | NCBI Datasets, `datasets` CLI 18.38.0 | 2026-10-11 |
| External phenotypes | NCBI Pathogen Detection AST Browser | 2026-10-11 |
| Resistance gene database | AMRFinderPlus 4.2.7 | database 2026-10-29.1 |

## Reproducibility notes

* Raw data is not modified manually.
* Processed datasets are generated through version-controlled scripts.
* Random seeds (0–4) and software dependencies are recorded.
* Dataset versions and download dates are documented.
* Large genomic files and the feature matrix are not committed.
* Intermediate and generated files are excluded through `.gitignore`.
* Models trained on CSR input treat absent k-mers as missing rather than zero;
  external scoring therefore passes `missing=0` explicitly when predicting from
  dense presence matrices. Omitting this returns a constant prediction and an
  AUROC of exactly 0.500.
* Scoring a fixed model is exactly reproducible; refitting is not. Report seed
  distributions rather than single fits wherever models are compared.

## Current status

* Fitting variance measured for all five externally testable models; the four
  that transfer are stable (sd 0.0006-0.0025), clindamycin is not (sd 0.0138)
* The 2x2 chi-square alternative tested and rejected on a pre-registered
  decision rule
* erm(B) identified by mecA-stratified ranking as a second causal clindamycin determinant missing from the pooled model; its association strengthens externally
* Erythromycin's secondary features identified as aminoglycoside resistance genes that travel with multidrug-resistant MRSA
* AMRFinderPlus run on all genomes: known genes explain about 92% of clindamycin resistance in both collections; erm(A), which travels with mecA in BV-BRC, and erm(T), absent from resistant training genomes, fill the gap
* Remaining: count conflicting phenotype records; label the stratified contigs at a 10% coverage threshold; possible expansion to additional antibiotics

## Limitations

* Methicillin phenotypes carry no MIC values and no recorded typing method for 78% of rows; performance should be read with that provenance in mind.
* Methicillin cannot be externally validated — NCBI holds no methicillin AST records — so it is the one antibiotic whose performance rests entirely on BV-BRC.
* External figures are ten-seed means. The four transferring models are stable
  (sd 0.0006-0.0025); clindamycin's sd is 0.0138, so its figure should always
  be quoted with its range (0.758-0.804).
* At 200,000 features, model-fitting variance (sd 0.014) exceeds the effect of excluding the confounded feature set. No intervention of that magnitude can be resolved by this design without many more fits.
* The paired bootstrap resamples external genomes while holding the fitted model fixed, so it reports evaluation variance only. Intervals produced by it earlier in this project understate total uncertainty and should not be read as significance tests on model comparisons.
* Lineage grouping uses k-mer similarity clusters as a proxy; formal MLST or core-genome typing has not been applied, and cluster boundaries depend on the chosen distance cutoff.
* Methicillin lineage-grouped splits are dominated by a single large cluster, so its cross-lineage estimate rests on effectively one replicate.
* The strain-grouped and lineage-grouped arms differ in design (70/10/20 with early stopping vs 80/20 with fixed trees), so the comparison between them is indicative rather than strictly like-for-like.
* Clindamycin is not usable as a predictor in either population and should be reported as a characterized failure rather than a performance figure.
* The external deduplication threshold (Jaccard ≥ 0.99 over a 20,000 k-mer panel) is a single chosen cutoff; 695 of 1,235 external genomes exceed 0.95, so the external collection is similar to the training collection at the panel level even after exclusion.
* *mecA* carriage is scored by presence of at least half the locus's k-mers; this threshold was not separately validated, and the two mecA contigs give near-identical carriage under it despite differing in per-k-mer presence.
* The relationship between best-feature odds ratio and external AUROC rests on five antibiotics and was identified after the results were in.
* The external evidence that erm(B) is causal rests on 8 carriers; its odds ratio interval spans 2.3 to 705.
* About 7-8% of clindamycin resistance (65 training and 24 external genomes) is explained by no catalogued gene; nearly all of it is also erythromycin-resistant, which suggests missed or fragmented erm calls rather than a new mechanism.
* The placement of erm(A) on Tn554 inside SCCmec, and the CC398 identity of the erm(T) lineage, are inferred from co-occurrence and the literature; neither was mapped on assemblies or confirmed by MLST.
* The ~92% coverage figure counts partial gene calls and includes *lnu*(A) and *lnu*(G), which do not associate with clindamycin resistance here (lnu(A) OR 0.7).
* Where BV-BRC held several records for one genome and antibiotic, the first qualifying record was kept; the number of conflicting records has not yet been counted.
* The feature-ranking score is a carrier-conditioned chi-square, not the 2x2 test with Yates' correction described in the AURA paper.
* Ciprofloxacin's external AUROC rests on 36 resistant genomes.

## Contributors

* Nada Elseifi — Graduate researcher
* Dr. Christina Boucher — Thesis advisor
* Ali (Mohammadali Serajian) — Research collaborator
