#!/usr/bin/env bash
set -euo pipefail

cd /Users/ayush.anand/projects/paper
mkdir -p logs

uv run python dino_training_ucf_mlp.py \
  --dataset-dir UCF101_Splits/UCF101TrainTestSplits-RecognitionTask \
  --video-dir UCF-101 \
  --batch-size 4 \
  --epochs 5 \
  2>&1 | tee logs/dino_ucf_mlp.out

uv run python blip_training_ucf_mlp.py \
  --dataset-dir UCF101_Splits/UCF101TrainTestSplits-RecognitionTask \
  --video-dir UCF-101 \
  --batch-size 4 \
  --epochs 5 \
  2>&1 | tee logs/blip_ucf_mlp.out

uv run python concat_mlp_ucf.py \
  --dataset-dir UCF101_Splits/UCF101TrainTestSplits-RecognitionTask \
  --video-dir UCF-101 \
  --batch-size 4 \
  --epochs 5 \
  2>&1 | tee logs/concat_ucf_mlp.out

uv run python dino_training_c100_mlp.py \
  --data-dir data \
  --download \
  --batch-size 32 \
  --epochs 20 \
  2>&1 | tee logs/dino_c100_mlp.out

uv run python blip_training_c100_mlp.py \
  --data-dir data \
  --download \
  --batch-size 32 \
  --epochs 20 \
  2>&1 | tee logs/blip_c100_mlp.out

uv run python concat_mlp_c100.py \
  --data-dir data \
  --download \
  --batch-size 32 \
  --epochs 20 \
  2>&1 | tee logs/concat_c100_mlp.out

uv run python dino_training_c10_mlp.py \
  --data-dir data \
  --download \
  --batch-size 32 \
  --epochs 20 \
  2>&1 | tee logs/dino_c10_mlp.out

uv run python blip_training_c10_mlp.py \
  --data-dir data \
  --download \
  --batch-size 32 \
  --epochs 20 \
  2>&1 | tee logs/blip_c10_mlp.out

uv run python concat_mlp_c10.py \
  --data-dir data \
  --download \
  --batch-size 32 \
  --epochs 20 \
  2>&1 | tee logs/concat_c10_mlp.out
