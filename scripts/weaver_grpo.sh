#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=4,5
export MAIN_PROCESS_PORT=29508
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Dataset configs
DATASET_NAME="kodcode"  # options: gsm8k, gpqa, kodcode, triviaqa

# MemGen configs
TRAIN_METHOD="grpo"    # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

GROUP_SIZE=8

LOAD_MODEL_PATH="MemGen-Models/Qwen2.5-1.5B-Instruct/kodcode/weaver-sft/pn=1_pl=8_in=0_il=8/model"

# train
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active False \
    datasets.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.grpo.max_completion_length 512 \
    run.weaver.grpo.num_train_epochs 1 \
    run.weaver.grpo.per_device_train_batch_size ${GROUP_SIZE} \
    run.weaver.grpo.per_device_eval_batch_size ${GROUP_SIZE} \
    run.weaver.grpo.num_generations ${GROUP_SIZE} \
    run.weaver.grpo.gradient_accumulation_steps 1 \
