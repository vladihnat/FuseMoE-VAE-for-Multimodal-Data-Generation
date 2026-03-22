#!/bin/bash
# Run FuseMoE training on preprocessed MIMIC-IV demo data (time-series only mode).
# This uses modeltype='TS' since the demo dataset lacks text, CXR, and ECG data.
#
# Usage:
#   conda activate fusemoe_research
#   bash run_demo_ts.sh

cd "$(dirname "$0")"

python src/scripts/main_mimiciv.py \
    --modeltype "TS" \
    --num_train_epochs 5 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --ts_learning_rate 0.0004 \
    --txt_learning_rate 0.00002 \
    --embed_dim 64 \
    --layers 2 \
    --num_heads 4 \
    --hidden_size 256 \
    --kernel_size 1 \
    --task "ihm-48-cxr-notes-ecg" \
    --file_path "Data/ihm" \
    --tt_max 48 \
    --embed_time 64 \
    --num_of_notes 5 \
    --max_length 512 \
    --num_labels 2 \
    --num_modalities 1 \
    --num_of_experts 4 \
    --top_k 2 \
    --gating_function "softmax" \
    --router_type "joint" \
    --cross_method "moe" \
    --seed 42 \
    --num_update_bert_epochs 0 \
    --bertcount 0 \
    --reg_ts \
    --irregular_learn_emb_ts \
    --debug \
    --output_dir "output_demo_ts"
