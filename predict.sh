#!/usr/bin/env bash
set -euo pipefail

# ===========================================
# Predict one checkpoint or all .model files
# ===========================================
# Examples:
# PREDICT_MODEL="train_aug"                     -> every .model in that folder
# PREDICT_MODEL="no_aug"                        -> every .model in that folder
# PREDICT_MODEL="train_aug/best_R3_sigmoid_pw1_s0.15_tau4_lr0.0003_bz1.model"
PREDICT_MODEL="train_aug"    # default follows TRAIN_SET="train" in train.sh
PREDICT_SET="val"             # val / val_aug / test / test_aug
LABEL_SET="vehicle"             # all=32 classes; vehicle=24 classes (0..23)

GPU_ID=4
DATASET="carom"
SPLIT=1
SPLIT_DIR="./data/carom/splits"
# Keep older 32-class locations unchanged; separate 24-class results.
if [[ "$LABEL_SET" == "vehicle" ]]; then
  MODEL_ROOT="./models/$DATASET/split_$SPLIT/vehicle"
  RESULTS_ROOT="./results/$DATASET/split_$SPLIT/vehicle"
else
  MODEL_ROOT="./models/$DATASET/split_$SPLIT"
  RESULTS_ROOT="./results/$DATASET/split_$SPLIT"
fi

# Must match the architecture used when training (not stored in standard tag).
NUM_LAYERS_PG=11
NUM_LAYERS_R=10
NUM_F_MAPS=64
FEATURES_DIM=2048

# Checkpoint tag R3_sigmoid or R1_logits is detected automatically by main.py.
# If the filename has no standard tag, these are used instead.
DEFAULT_NUM_R=3
DEFAULT_REFINEMENT_INPUT="sigmoid"

STAGE_DIAGNOSTICS=false  # true / false

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f main.py ]]; then
  echo "ERROR: main.py not found in $ROOT_DIR" >&2
  exit 1
fi
if [[ ! -f "$SPLIT_DIR/${PREDICT_SET}.split${SPLIT}.bundle" ]]; then
  echo "ERROR: missing split bundle: $SPLIT_DIR/${PREDICT_SET}.split${SPLIT}.bundle" >&2
  exit 1
fi

# Select by relative folder/checkpoint, or by full path.
case "$PREDICT_MODEL" in
  /*|./*|../*) MODEL_TARGET="$PREDICT_MODEL" ;;
  *) MODEL_TARGET="$MODEL_ROOT/$PREDICT_MODEL" ;;
esac

if [[ -d "$MODEL_TARGET" ]]; then
  MODEL_SOURCE="$(basename -- "$MODEL_TARGET")"
  shopt -s nullglob
  CHECKPOINTS=("$MODEL_TARGET"/*.model)
  shopt -u nullglob
  if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "ERROR: no .model checkpoints in $MODEL_TARGET" >&2
    exit 1
  fi
elif [[ -f "$MODEL_TARGET" ]]; then
  MODEL_SOURCE="$(basename -- "$(dirname -- "$MODEL_TARGET")")"
  CHECKPOINTS=("$MODEL_TARGET")
else
  echo "ERROR: model file or folder does not exist: $MODEL_TARGET" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

DIAGNOSTICS_ARGS=()
if [[ "$STAGE_DIAGNOSTICS" == "true" ]]; then
  DIAGNOSTICS_ARGS+=(--stage_diagnostics)
fi

echo "GPU: $GPU_ID"
echo "Label set: $LABEL_SET"
echo "Prediction split: $PREDICT_SET"
echo "Model selection:  $MODEL_TARGET"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Length policy: min(Feature, GT) in model.py"

for checkpoint in "${CHECKPOINTS[@]}"; do
  checkpoint_name="$(basename -- "$checkpoint" .model)"
  out_dir="$RESULTS_ROOT/$PREDICT_SET/$MODEL_SOURCE/$checkpoint_name"
  mkdir -p "$out_dir"
  echo
  echo "============================="
  echo "Predicting: $checkpoint"
  echo "Output:     $out_dir"
  echo "============================="

  python main.py \
    --action predict \
    --label_set "$LABEL_SET" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --split_dir "$SPLIT_DIR" \
    --predict_set "$PREDICT_SET" \
    --predict_model_path "$checkpoint" \
    --results_dir "$out_dir" \
    --features_dim "$FEATURES_DIM" \
    --num_layers_PG "$NUM_LAYERS_PG" \
    --num_layers_R "$NUM_LAYERS_R" \
    --num_f_maps "$NUM_F_MAPS" \
    --num_R "$DEFAULT_NUM_R" \
    --refinement_input "$DEFAULT_REFINEMENT_INPUT" \
    "${DIAGNOSTICS_ARGS[@]}"
done

echo
echo "Done. Results under: $RESULTS_ROOT/$PREDICT_SET/$MODEL_SOURCE/"
