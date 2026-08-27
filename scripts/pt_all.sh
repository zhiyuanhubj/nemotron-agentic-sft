#!/usr/bin/env bash
# Pretokenize the V4 ablation splits. Run under docker so the tokenizer matches training.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}
MODEL=${NEMOTRON_MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
DATA_DIR=${DATA_DIR:-$ROOT/data}

for g in all swe_only tmax_only; do
  for part in train val; do
    inp=$DATA_DIR/${g}_${part}_clean.jsonl
    out=$DATA_DIR/${g}_${part}_tok.jsonl
    [ -f "$inp" ] || { echo "skip missing $inp"; continue; }
    echo "=== ${g}_${part}"
    docker run --rm --network host \
      -v "$ROOT:$ROOT" -v "$DATA_DIR:$DATA_DIR" \
      --entrypoint python3 "$IMAGE" \
      "$ROOT/scripts/pretokenize.py" --model "$MODEL" --inp "$inp" --out "$out"
  done
done
