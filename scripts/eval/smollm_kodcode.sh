#!/bin/bash

export DEBUG_MODE=true  
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29508
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

REASONER_MODEL="HuggingFaceTB/SmolLM3-3B"
WEAVER_MODEL="HuggingFaceTB/SmolLM3-3B"   
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_ACTIVE=False

DATASET_NAME="kodcode"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=4
INFERENCE_LATENTS_LEN=4

BATCH_SIZE=4

LOAD_MODEL_PATH="MemGen/SmolLM3-3B/kodcode/weaver-sft/pn=1_pl=4_in=5_il=4"

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
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.temperature 0.0 \
    run.interaction.max_response_length 1024 \