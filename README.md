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
* Duplicate genome IDs identified and handled
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

Genome IDs from the cleaned phenotype matrix were used to request nucleotide FASTA files from BV-BRC, downloaded on HiPerGator using the BV-BRC Browser Tools. Genome files use the `.fna` extension; note that gerbil (inside KMX) rejects that extension, so `.fasta` symlinks were created.

Large `.fna` files and the feature matrix remain on HiPerGator and are not committed here. This repository contains genome ID lists, processing and modeling code, documentation, summary statistics, and small tabular results.

## Feature matrix

Built with KMX (github.com/M-Serajian/KMX): **4,610 genomes × 11,351,471 k-mers**, k=31, min=5, max=3000, 90.6% sparse, stored as CSR on HiPerGator at `/blue/boucher/nelseifi/kmx_output/`. Chi-square ranking uses presence/absence rather than raw counts, so that multi-copy repetitive k-mers do not dominate the ranking.

## Repository structure

```text
staph-aureus-amr-prediction/
├── README.md
├── data/                          # phenotype tables, ID lists, results JSONs
│   ├── FINAL_staph_aureus_AMR_phenotypes.csv
│   ├── FILTERED_staph_aureus_AMR_phenotypes.csv
│   ├── phenotypes_matrix_aligned.csv
│   ├── genome_sources.csv         # genome -> source study mapping
│   ├── lineage_clusters.csv       # genome -> k-mer similarity cluster
│   ├── missing_ids.txt
│   ├── staph_results/             # random-split results, all feature counts
│   ├── staph_results_source/      # held-out-study results
│   ├── staph_results_source_cal/  # threshold-calibrated variant
│   ├── staph_results_models/      # five-model comparison
│   └── staph_results_lineage/     # lineage-grouped results
└── src/
    ├── clean_amr.py               # phenotype cleaning
    ├── make_one_final_csv.py      # wide phenotype matrix
    ├── make_amr_stats.py          # summary statistics
    ├── run_kmx.sh                 # k-mer matrix (GPU Slurm job)
    ├── make_splits.py             # strain-grouped 70/10/20 splits
    ├── chi2_rank.py               # chi-square feature ranking
    ├── train_pilot.py             # XGBoost training and evaluation
    ├── model_compare.py           # five-model comparison
    ├── kmer_interpret.py          # top-feature extraction
    ├── source_holdout.py          # cross-study evaluation
    ├── source_holdout_cal.py      # threshold-calibrated variant
    ├── lineage_cluster.py         # k-mer similarity clustering
    ├── make_splits_lineage.py     # lineage-grouped 80/20 splits
    └── train_lineage.py           # training on lineage-grouped splits
```

## Data-processing workflow

1. Download the complete BV-BRC AMR phenotype table as CSV.
2. Count unique BV-BRC genome IDs.
3. Filter records to *Staphylococcus aureus*.
4. Retain laboratory-based phenotype evidence only.
5. Standardize phenotype labels.
6. Reshape into one row per genome, one column per antibiotic.
7. Assign `None` where an antibiotic was not tested.
8. Calculate resistant, susceptible, and untested counts per antibiotic.
9. Select antibiotics meeting the sample-size and resistance-ratio criteria.
10. Download the corresponding `.fna` genome sequences on HiPerGator.
11. Build the k-mer feature matrix with KMX.
12. Align phenotype rows to matrix row order (`phenotypes_matrix_aligned.csv`).
13. Create training, validation, and test partitions without genome or strain leakage.
14. Rank features by chi-square on training rows only.
15. Train and evaluate antibiotic-specific prediction models.

## Data-leakage prevention

Training, validation, and test sets are separated at the genome level; records representing the same genome never appear in multiple partitions. Splits are grouped by strain name so that repeated strain records stay within one partition, and each split is verified by assertion before use. All preprocessing — including chi-square feature selection — is fitted on training data only and applied unchanged to validation and test sets.

Partitions are 70/10/20, repeated over five fixed seeds (0–4), with class counts recorded for every split. A second, stricter split design groups genomes by k-mer similarity cluster (see *Lineage-grouped evaluation* below).

## Model evaluation

Models are assessed with accuracy, precision, recall, F1, ROC-AUC, precision–recall AUC, confusion matrices, and class-specific performance, reported as mean ± standard deviation across the five seeds.

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
used anywhere in the pipeline.

| Antibiotic | Top locus | Genetic context | R / S |
|---|---|---|---|
| methicillin | *mecA* (PBP2a) | SCCmec | 98.5% / 0.5% |
| oxacillin | *mecA* | SCCmec | 90.4% / 7.3% |
| cefoxitin | *mecA* | SCCmec | 88.0% / 12.9% |
| ciprofloxacin | *gyrA* S84L, *grlA* S80F | chromosomal point mutations | 86.1% / 3.4% |
| erythromycin | *erm*(C) | 2,473 bp plasmid | 48.1% / 2.4% |
| clindamycin | *mecA* | SCCmec | 82.2% / 20.5% |

**Contig length distinguishes mechanism class.** Ciprofloxacin's longest contig
is 75 bp while the others reach 2,500–2,800 bp. A point mutation alters exactly
*k* k-mers, which assemble into a 2*k*−1 bp window; an acquired gene contributes
k-mers across its full length. Contig length therefore diagnoses point-mutation
versus acquired-gene resistance without annotation.

**Both alleles of the quinolone-resistance mutations were recovered.** The four
top ciprofloxacin contigs form two pairs differing at a single nucleotide:
*gyrA* S84L (86.1% R / 3.4% S) with wild-type Ser84 (7.8% R / 95.5% S), and
*grlA* S80F (81.4% R / 4.7% S) with wild-type Ser80 (4.3% R / 91.0% S). Each is
61 bp from exactly 31 k-mers, the arithmetic signature of a SNP.

**Erythromycin's determinant is plasmid-borne.** Its top contig closes on
itself with a 30 bp overlap, giving a 2,473 bp circular element carrying a
244-aa ORF with the GSGKG (SAM-binding) and NIPY (N6-adenine methyltransferase)
motifs, confirmed by blastx as Erm(C) at 100% identity.

**Clindamycin's top features are *mecA*, not an MLSb determinant.** Its two
leading contigs are the same *mecA*-containing loci ranked first by the
beta-lactams. The *erm*(C) plasmid does not appear among its features at all.
The model predicts clindamycin resistance through linkage to SCCmec rather than
through a causal determinant, which is consistent with its being the one
antibiotic that retains a lineage penalty under lineage-grouped evaluation.

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

Single-determinant mechanisms saturate by ~1,000 features. The MLSb antibiotics (erythromycin, clindamycin) require roughly 50× more, consistent with resistance distributed across multiple *erm* genes rather than one determinant. Clindamycin plateaus near 200,000.

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

Two effects separate out. First, an insufficient feature budget can masquerade as lineage sensitivity: at 10,000 features the MLSb antibiotics appear to depend on population structure, but erythromycin fully recovers at 50,000 (0.954) and exceeds its strain-grouped score at 200,000 (0.963), with seed variance falling from ±0.054 to ±0.014. Second, clindamycin retains a residual lineage effect after the feature budget is corrected (0.857 at 50k, plateauing, versus 0.892 strain-grouped, seed sd ±0.071), consistent with inducible MLSb resistance where *erm* presence does not fully determine phenotype.

Practical consequence: feature counts should be set per antibiotic, with ~50,000 as a floor for MLSb drugs rather than a single global value.

### Cross-study generalization

Phenotypes derive from ~15 source studies with sharply differing resistance prevalence; source study alone predicts the label at 72–85% accuracy. Held-out-study evaluation shows discrimination largely survives (AUROC 0.82–0.99), but decision thresholds do not transfer. Calibrating thresholds on other training studies did not help and degraded F1 for four of six antibiotics (`source_holdout_cal.py`, documented negative result). AUROC and AUPRC are the appropriate cross-study metrics.

## Reproducibility notes

* Raw data is not modified manually.
* Processed datasets are generated through version-controlled scripts.
* Random seeds (0–4) and software dependencies are recorded.
* Dataset versions and download dates are documented.
* Large genomic files and the feature matrix are not committed.
* Intermediate and generated files are excluded through `.gitignore`.

## Current status

* Phenotype processing, genome retrieval, and feature-matrix construction complete
* Leakage-controlled splits and chi-square feature selection complete
* Six antibiotic-specific models trained and evaluated across five classifiers
* Known resistance mechanisms recovered and verified (*mecA*, *gyrA*, *grlA*)
* Feature-count sensitivity, cross-study robustness, and lineage-grouped
  evaluation complete
* Remaining: characterization of high-ranking features not matching known
  determinants, possible expansion to additional antibiotics, external validation

## Limitations

* Methicillin phenotypes carry no MIC values and no recorded typing method for 78% of rows; performance should be read with that provenance in mind.
* Lineage grouping uses k-mer similarity clusters as a proxy; formal MLST or core-genome typing has not been applied, and cluster boundaries depend on the chosen distance cutoff.
* Methicillin lineage-grouped splits are dominated by a single large cluster, so its cross-lineage estimate rests on effectively one replicate.
* The strain-grouped and lineage-grouped arms differ in design (70/10/20 with early stopping vs 80/20 with fixed trees), so the comparison between them is indicative rather than strictly like-for-like.
* External validation on an independent database or time period has not yet been performed.

## Contributors

* Nada Elseifi — Graduate researcher
* Dr. Christina Boucher — Thesis advisor
* Ali (Mohammadali Serajian) — Research collaborator
