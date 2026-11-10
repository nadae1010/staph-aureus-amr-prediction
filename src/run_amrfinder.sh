#!/bin/bash
#SBATCH --job-name=amrfinder
#SBATCH --account=boucher
#SBATCH --qos=boucher-b
#SBATCH --array=0-49%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/amrfinder_%A_%a.log

# Runs NCBI AMRFinderPlus (4.2.7, database 2026-10-29.1) on every genome in
# amrfinder/genomes.txt: the 4,610 training genomes (genome_fasta_list_fasta.txt,
# in matrix row order) followed by the external NCBI assemblies.
# 50 array tasks; task t handles lines t, t+50, t+100, ...
# Output: amrfinder/out/<first 16 hex of md5(path)>.tsv, one per genome.
# Safe to resubmit: genomes that already have output are skipped.

module load amrfinderplus
cd ~/staph_amr_project

LIST=amrfinder/genomes.txt
OUT=amrfinder/out
DB=$HOME/amrfinder_db/latest
NTASK=50
mkdir -p "$OUT"

amrfinder -V
i=0; ran=0; failed=0
while IFS= read -r f; do
  if (( i % NTASK == SLURM_ARRAY_TASK_ID )); then
    o="$OUT/$(printf '%s' "$f" | md5sum | cut -c1-16).tsv"
    if [[ ! -s "$o" ]]; then
      if amrfinder -n "$f" -O Staphylococcus_aureus --plus --threads 4 \
           -d "$DB" --name "$f" -o "$o.tmp" < /dev/null; then
        mv "$o.tmp" "$o"; ran=$((ran+1))
      else
        echo "FAILED: $f"; rm -f "$o.tmp"; failed=$((failed+1))
      fi
    fi
  fi
  i=$((i+1))
done < "$LIST"
echo "task $SLURM_ARRAY_TASK_ID finished: $ran run, $failed failed"
