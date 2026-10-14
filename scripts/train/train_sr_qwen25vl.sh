#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# This backbone was released against Transformers 4.50.0.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/geothinker/geothinker_qwen25vl}"
CACHE_DIR="${CACHE_DIR:-./cache}"
DATASETS="${DATASETS:-llava_hound_64k_32frame,spar_234k,vsi_590k_32frame,vlm3r_vsi_205k_32frames,vlm3r_vst_132k_32frames,phygames_140k,mindcube_10k,cambrian_s_3m_subset_32frame}"
LR="${LR:-1e-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WORLD_SIZE="${WORLD_SIZE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((TOTAL_BATCH_SIZE / NPROC_PER_NODE / WORLD_SIZE))}"

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node="$NPROC_PER_NODE" \
            --nnodes="$WORLD_SIZE" \
            --node_rank="$NODE_RANK" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            src/qwen_vl/train/train_qwen.py \
            --geometry_method geothinker \
            --model_name_or_path "$MODEL_PATH" \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use "$DATASETS" \
            --output_dir "$OUTPUT_DIR" \
            --cache_dir "$CACHE_DIR" \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
            --learning_rate "$LR" \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 12800 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 50 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_geometry_encoder true \
            --geometry_encoder_type "$GEOMETRY_ENCODER_TYPE" \
            --geometry_encoder_path "$GEOMETRY_ENCODER_PATH" \
            --feature_fusion_method "zero" \
            --geo_cross_attn True \
            --geo_inject_version "v2_flash" \
            --geo_importance_gate True \
            --depart_smi_token True \
            > "${OUTPUT_DIR}/train.log" 2>&1
