#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1


REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

DATASET_NAME="gsm8k"

TRAIN_METHOD="sft"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=3
PROMPT_LATENTS_LEN=8
INFERENCE_LATENTS_LEN=8

BATCH_SIZE=1

LOAD_MODEL_PATH=null


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
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.bf16 True \
    run.weaver.sft.gradient_accumulation_steps 1 \
