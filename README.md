# Staphylococcus aureus Antimicrobial Resistance Prediction

## Project overview

This project develops a machine-learning pipeline for predicting antimicrobial resistance (AMR) in *Staphylococcus aureus* using genomic sequences and laboratory-based phenotype data from the Bacterial and Viral Bioinformatics Resource Center (BV-BRC).

The project is being completed as part of a master’s thesis under the supervision of Dr. Christina Boucher at the University of Florida.

## Research objective

The goal is to train and evaluate models that predict whether a *Staphylococcus aureus* genome is resistant or susceptible to individual antibiotics. The project will also examine model generalizability, phenotype-label consistency, class imbalance, interpretability, and prevention of data leakage.

## Data-selection criteria

The dataset was prepared using the following criteria:

* Organism: *Staphylococcus aureus* only
* Source: BV-BRC
* More than 1,500 samples per selected antibiotic
* Resistant phenotype ratio of at least 30%
* Only laboratory-based phenotype evidence retained
* Data downloaded without filtering and cleaned afterward
* Duplicate genome IDs identified and handled
* Untested genome–antibiotic combinations recorded as `None`

## Phenotype dataset

The cleaned phenotype matrix contains one row per genome and uses the following structure:

1. `genome_id`
2. `genome_name`
3. `strain_name`
4. One phenotype column per antibiotic

Phenotype values are standardized as:

* `Resistant`
* `Susceptible`
* `None` — antibiotic not tested for that genome

The cleaned dataset currently contains approximately **4,944 unique genomes**.

Examples of the most frequently reported antibiotics include:

* Methicillin: 2,047 tested genomes; approximately 56.7% resistant
* Cefoxitin: 2,096 tested genomes; approximately 63.0% resistant

Additional antibiotic-level statistics will be maintained with the processed dataset.

## Genome sequence data

Genome IDs from the cleaned phenotype matrix were used to request corresponding nucleotide FASTA files from BV-BRC. Genome files use the `.fna` extension.

Genome downloads were performed on the University of Florida HiPerGator system using the BV-BRC Browser Tools.

Large `.fna` files should remain on HiPerGator and should not be committed directly to GitHub. This repository should contain the genome ID list, download scripts, processing code, documentation, summary statistics, and small tabular files needed to reproduce the workflow.

## Planned repository structure

```text
staph-aureus-amr-prediction/
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── genome_ids/
├── scripts/
│   ├── phenotype_processing/
│   ├── genome_download/
│   └── machine_learning/
├── results/
│   ├── dataset_statistics/
│   ├── model_performance/
│   └── figures/
├── docs/
└── requirements.txt
```

## Data-processing workflow

1. Download the complete BV-BRC AMR phenotype table as CSV.
2. Count unique BV-BRC genome IDs.
3. Filter records to *Staphylococcus aureus*.
4. Retain laboratory-based phenotype evidence only.
5. Standardize phenotype labels.
6. Reshape the data into one row per genome.
7. Create one phenotype column per antibiotic.
8. Assign `None` where an antibiotic was not tested.
9. Calculate resistant, susceptible, and untested counts for each antibiotic.
10. Select antibiotics meeting the sample-size and resistance-ratio criteria.
11. Create a text file containing the selected genome IDs.
12. Download the corresponding `.fna` genome sequences on HiPerGator.
13. Create training, validation, and test partitions without genome or strain leakage.
14. Train and evaluate antibiotic-specific prediction models.

## Data-leakage prevention

Training, validation, and test sets must be separated at the genome level. Records representing the same genome must never appear in multiple partitions. Strain-level grouping will also be evaluated so that closely related or repeated strain records do not create an artificially optimistic estimate of model performance.

All preprocessing decisions must be fitted using training data only and then applied unchanged to the validation and test sets.

## Model evaluation

Models will be assessed using metrics appropriate for potentially imbalanced phenotype data, including:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC
* Precision–recall AUC
* Confusion matrix
* Class-specific performance

External or temporally separated validation will be considered where the available data permit it.

## Reproducibility notes

* Raw data should not be modified manually.
* Processed datasets should be generated through version-controlled scripts.
* Random seeds and software dependencies should be recorded.
* Dataset versions and download dates should be documented.
* Large genomic files and sensitive or restricted data must not be committed.
* Intermediate and generated files should be excluded through `.gitignore`.

## Current status

* BV-BRC phenotype data downloaded
* *Staphylococcus aureus* records filtered
* Laboratory-based phenotype evidence retained
* Machine-learning-ready phenotype matrix created
* Approximately 4,944 genome IDs identified
* Genome sequence download initiated on HiPerGator
* Dataset documentation and repository organization in progress
* Machine-learning experiments not yet finalized

## Contributors

* Nada Elseifi — Graduate researcher
* Dr. Christina Boucher — Thesis advisor
* Ali — Research collaborator
