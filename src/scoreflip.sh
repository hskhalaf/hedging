#!/bin/bash
#SBATCH -c 2
#SBATCH -p gpu
#SBATCH --mem=160000
#SBATCH -t 0-24:00:00
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

SIZE=10k

for flip in flip00 flip25; do
  mkdir -p score/${SIZE}/${flip} logs/${SIZE}/${flip}
done

models=(models/*)
filtered_models=()

for model in "${models[@]}"; do
  name=$(basename "$model")
  if [[ "$name" =~ ^flip(00|25)-rm-pythia-44m_${SIZE}_seed[123]$ ]]; then
    filtered_models+=("$model")
  fi
done

for model_path in "${filtered_models[@]}"; do
  name=$(basename "$model_path")
  flip=${name%%-*}
  out=score/${SIZE}/${flip}/${name}.json
  logo=logs/${SIZE}/${flip}/${name}.out
  loge=logs/${SIZE}/${flip}/${name}.err

  echo "[START] $name"
  python test.py \
    --model_path "$model_path" \
    --output_file "$out" \
    --max_rows 900 \
    --batch_size 1400 \
    --max_answers 12600 \
    >"$logo" 2>"$loge"
  echo "[ DONE]  $name"
done


