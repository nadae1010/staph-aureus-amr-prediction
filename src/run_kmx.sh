# KMX k-mer matrix job for S. aureus AMR project (HiPerGator)
# Input: genome_fasta_list.txt (4,610 BV-BRC .fna paths)
# Params per Ali: k=31, --min 5, --max 3000
# Submit: sbatch run_kmx.sh
#!/bin/bash
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

cd ~/staph_amr_project/KMX

python KMX.py \
    -l ~/staph_amr_project/BV-BRC-BrowserTools/genome_fasta_list.txt \
    -k 31 \
    -t /blue/boucher/nelseifi/kmx_tmp \
    -o ~/staph_amr_project/kmx_output \
    --min 5 \
    --max 3000
