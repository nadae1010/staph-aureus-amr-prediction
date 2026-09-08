#!/bin/bash
# KMX k-mer matrix job for S. aureus AMR project (HiPerGator)
# Input: genome_fasta_list_fasta.txt (4,610 symlinks with .fasta extension
#   pointing to BV-BRC .fna files in /home/m.serajian/share/BV-BRC/staph/
#   — gerbil rejects the .fna extension, so links were created in
#   /blue/boucher/nelseifi/staph_fasta_links/)
# Params per Ali: k=31, min=5, max=3000 (note: single-dash -min/-max)
# KVIKIO_COMPAT_MODE=ON required on HiPerGator (cuFile "Operation not
#   permitted" error otherwise)
# Submit: sbatch run_kmx.sh
#SBATCH --job-name=kmx_staph_k31
#SBATCH --account=boucher
#SBATCH --qos=boucher
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=kmx_%j.log

module load conda
conda activate KMX-env
export KVIKIO_COMPAT_MODE=ON

cd ~/staph_amr_project/KMX

python KMX.py \
    -l ~/staph_amr_project/genome_fasta_list_fasta.txt \
    -k 31 \
    -t /blue/boucher/nelseifi/kmx_tmp \
    -o ~/staph_amr_project/kmx_output \
    -min 5 \
    -max 3000
