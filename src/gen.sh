#!/bin/bash
#SBATCH -c 2
#SBATCH -p  
#SBATCH --mem=80000
#SBATCH -t 0-4:00:00
#SBATCH -o gen_%j.out
#SBATCH -e gen_%j.err

source ~/.bashrc
mamba activate llmenv

N_TRIALS=256
TOTAL_SAMPLES=60000

python gen.py \
  --model_name "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr" \
  --n_trials       $N_TRIALS \
  --total_samples $TOTAL_SAMPLES \
  --batch_size       1 \
  --chunk_size     256
