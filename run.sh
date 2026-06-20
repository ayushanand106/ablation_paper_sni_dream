#!/usr/bin/env bash
set -euo pipefail

cd /Users/ayush.anand/projects/paper
mkdir -p logs
aria2c -x 16 -s 16 https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --check-certificate=false

unrar x UCF101.rar && rm UCF101.rar

git clone https://github.com/brahmesh001/UCF101_Splits

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
