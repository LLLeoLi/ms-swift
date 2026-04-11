#!/bin/bash
# Megatron SFT for Qwen3-Coder-30B-A3B-Instruct
# MoE model: 30B total, 3B active parameters
# Requires: 8 GPUs (80GiB each), adjust parallelism as needed

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model /ssddata/lihao/projects/models/Qwen3-Coder-30B-A3B-Instruct \
    --dataset /ssddata/lihao/projects/task-sync/data/sft.jsonl \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --num_train_epochs 3 \
    --finetune true \
    --tuner_type full \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_length 65536 \
    --save_safetensors true \
    --output_dir megatron_output/Qwen3-Coder-30B-A3B-Instruct \
    --save_steps 50 \
    --eval_steps 50 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    --padding_free true \
    --packing true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true
