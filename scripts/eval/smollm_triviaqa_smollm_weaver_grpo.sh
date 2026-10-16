#!/bin/bash
# SmolLM3-3B Reasoner, HuggingFaceTB/SmolLM3-3B Weaver, inactive Trigger.

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29508

# Count the visible GPUs automatically.
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

REASONER_MODEL="HuggingFaceTB/SmolLM3-3B"
WEAVER_MODEL="HuggingFaceTB/SmolLM3-3B"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_ACTIVE=False

DATASET_NAME="triviaqa"

MAX_PROMPT_AUG_NUM=8
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=4
INFERENCE_LATENTS_LEN=4

LORA_TARGET_MODULES="[q_proj,v_proj]"

BATCH_SIZE=16

LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-hf_upload/SmolLM3-3B/triviaqa/weaver-smollm3-3b/weaver-grpo/pn=8_pl=4_in=0_il=4/model}"

python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=${NUM_GPUS} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.attn_implementation sdpa \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.lora_config.target_modules ${LORA_TARGET_MODULES} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.lora_config.target_modules ${LORA_TARGET_MODULES} \
    model.trigger.active ${TRIGGER_ACTIVE} \
    run.mode evaluate \
    run.interaction.batch_size ${BATCH_SIZE} \
    run.interaction.temperature 0.0 \
    run.interaction.max_turns 5 \
    run.interaction.max_prompt_length 16384 \
    run.interaction.max_response_length 2048 \
    run.interaction.max_obs_length 512
