#!/usr/bin/env bash
# Runs every experiment in the project for each seed, then builds the results table.
# Usage: bash scripts/run_experiments.sh /path/to/BreaKHis [epochs]
set -euo pipefail

DATA="${1:?usage: bash scripts/run_experiments.sh /path/to/BreaKHis [epochs]}"
EPOCHS="${2:-15}"
SEEDS="${SEEDS:-0 1 2}"

run() { python train.py --data-dir "$DATA" --epochs "$EPOCHS" "$@"; }

for s in $SEEDS; do
  # RQ1: small vs large model, patient-level split
  run --model mobilenet_v2 --seed "$s" --run-name e1_mobilenet_s"$s"
  run --model resnet50     --seed "$s" --run-name e2_resnet50_s"$s"

  # RQ2: class imbalance
  run --model mobilenet_v2 --loss weighted       --seed "$s" --run-name e3_weighted_s"$s"
  run --model mobilenet_v2 --sampler oversample  --seed "$s" --run-name e4_oversample_s"$s"

  # RQ3: random image-level split, to measure the inflation
  run --model mobilenet_v2 --split image --seed "$s" --run-name e5_image_split_s"$s"

  # RQ4: our improvement, stain augmentation and distillation from the ResNet-50 teacher
  run --model mobilenet_v2 --stain-aug --seed "$s" --run-name e6_stain_s"$s"
  run --model mobilenet_v2 --teacher checkpoints/e2_resnet50_s"$s".pt --seed "$s" --run-name e7_distill_s"$s"
  run --model mobilenet_v2 --teacher checkpoints/e2_resnet50_s"$s".pt --stain-aug --seed "$s" --run-name e8_distill_stain_s"$s"
done

python collect_results.py
