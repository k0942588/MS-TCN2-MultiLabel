#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# MS-TCN2 MultiLabel training configuration
# ==========================================
GPU_ID=4
DATASET="carom"
SPLIT=1
SPLIT_DIR="./data/carom/splits"

TRAIN_SET="train"       # train / train_aug
VAL_SET="val"           # val / val_aug

NUM_EPOCHS=300
NUM_LAYERS_PG=11
NUM_LAYERS_R=10
NUM_R=3
NUM_F_MAPS=64
LR=0.0007
POS_WEIGHT=1
BATCH_SIZE=1
REFINEMENT_INPUT="sigmoid"

# Save train -> no_aug, train_aug -> train_aug so predict.sh can find them.
if [[ "$TRAIN_SET" == "train_aug" ]]; then
  MODEL_GROUP="train_aug"
else
  MODEL_GROUP="no_aug"
fi
MODEL_DIR="./models/$DATASET/split_$SPLIT/$MODEL_GROUP"

# Run from repository root (same folder as main.py).
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f main.py ]]; then
  echo "ERROR: main.py not found in $ROOT_DIR" >&2
  exit 1
fi
for bundle in \
  "$SPLIT_DIR/${TRAIN_SET}.split${SPLIT}.bundle" \
  "$SPLIT_DIR/${VAL_SET}.split${SPLIT}.bundle"; do
  if [[ ! -f "$bundle" ]]; then
    echo "ERROR: missing split bundle: $bundle" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "GPU: $GPU_ID"
echo "Train: ${TRAIN_SET}.split${SPLIT}.bundle"
echo "Val:   ${VAL_SET}.split${SPLIT}.bundle"
echo "Length policy: min(Feature, GT) in batch_gen.py"
echo "Model output: $MODEL_DIR"

python main.py \
  --action train \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --split_dir "$SPLIT_DIR" \
  --model_dir "$MODEL_DIR" \
  --train_set "$TRAIN_SET" \
  --val_set "$VAL_SET" \
  --num_epochs "$NUM_EPOCHS" \
  --num_layers_PG "$NUM_LAYERS_PG" \
  --num_layers_R "$NUM_LAYERS_R" \
  --num_R "$NUM_R" \
  --num_f_maps "$NUM_F_MAPS" \
  --lr "$LR" \
  --bce_pos_weight "$POS_WEIGHT" \
  --bz "$BATCH_SIZE" \
  --refinement_input "$REFINEMENT_INPUT" \
  --save_best_only \
  --stage_diagnostics
