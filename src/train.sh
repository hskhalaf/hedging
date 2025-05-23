#!/bin/bash
#SBATCH -c 2                            # Number of CPU cores
#SBATCH --mem=160000                    # Total memory (MB)
#SBATCH -t 0-7:00:00                   # Walltime
#SBATCH -o rm_pythia44m00%j.out          # STDOUT → filename with jobid
#SBATCH -e rm_pythia44m00%j.err          # STDERR → filename with jobid
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

CONFIGS=(rm-pythia-44m_80k)

# loop over seeds 1–4 and config variants
for SEED in 1 2 3 4; do
  for CFG in "${CONFIGS[@]}"; do
    accelerate launch \
      --config_file configs/accelerate_config_simple.yaml \
      src/reward_modeling/training/trainer_rm.py \
      --configs defaults_rm $CFG \
      --per_device_eval_batch_size 128 \
      --per_device_train_batch_size 128 \
      --rng_seed $SEED
  done
done

