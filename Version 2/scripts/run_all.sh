#!/usr/bin/env bash
# Full pipeline orchestration, matching the invocation the original
# implementation brief specified. Requires: network access, torch,
# transformers, datasets, sentence-transformers, a GPU (or patience on
# CPU), and the language list you actually want to run configured in
# configs/default_config.yaml.
#
# THIS HAS NOT BEEN EXECUTED. It was written in a sandbox with no
# network access, no GPU, and no torch/transformers installed -- see
# README.md's "Why this is scoped the way it is" section. Every command
# below has been individually syntax-checked and its non-torch-dependent
# logic unit-tested (241 tests as of this writing, all passing), but the
# full chain has never run end-to-end. Expect to debug integration
# issues on first real run -- that is normal for a pipeline this size
# and is exactly what running it yourself would surface.
#
# Usage:
#   bash scripts/run_all.sh [language_code]
# Defaults to "en" if no language is given, since that's the only
# language with a real (if minimal) demographic dictionary
# (fairness/demographic_dictionaries.py) wired up out of the box.

set -euo pipefail

LANG_CODE="${1:-en}"
OUT_DIR="outputs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "=== [1/8] Installing dependencies ==="
pip install -r requirements.txt

echo "=== [2/8] Downloading and building sentiment data (lang=$LANG_CODE) ==="
python datasets/build_sentiment.py --languages "$LANG_CODE" \
    --out-dir "data/processed/sentiment"

echo "=== [3/8] Building counterfactual pairs ==="
python datasets/build_counterfactual_pairs.py \
    --input "data/processed/sentiment/train.jsonl" \
    --output "data/processed/sentiment/train_pairs.jsonl" \
    --language "$LANG_CODE"

echo "=== [4/8] Training ADAPT-BTS ==="
python scripts/train.py \
    --model-config configs/mt5.yaml --task-config configs/sentiment.yaml \
    --set checkpoint.output_dir="$OUT_DIR/checkpoints/adapt_bts"

echo "=== [5/8] Training baselines ==="
python baselines/train_all_baselines.py \
    --model-config configs/mt5.yaml --task-config configs/sentiment.yaml \
    --baselines mt5_ft mfc csd madl grad_unlearn magnet

echo "=== [6/8] Evaluating all methods ==="
python scripts/evaluate.py \
    --model-config configs/mt5.yaml --task-config configs/sentiment.yaml \
    --checkpoint "$OUT_DIR/checkpoints/adapt_bts/final_model.pt" \
    --split test
# NOTE: repeat this evaluate.py call once per baseline checkpoint too --
# left as separate invocations rather than looped here since each
# baseline's checkpoint path and (for MFC/MADL) model wrapper differ;
# see baselines/train_all_baselines.py for where each gets saved.

echo "=== [7/8] Generating tables ==="
python scripts/reproduce_tables.py \
    --reports "ADAPT-BTS=$OUT_DIR/adapt_bts_eval.json" \
    --out-dir "$OUT_DIR/tables"

echo "=== [8/8] Generating figures ==="
python scripts/reproduce_figures.py \
    --reports "ADAPT-BTS=$OUT_DIR/adapt_bts_eval.json" \
    --highlight "ADAPT-BTS" \
    --out-dir "$OUT_DIR/figures"

echo "Done. Outputs in $OUT_DIR"
