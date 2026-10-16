#!/bin/bash
# Train Weaver SFT with a SmolLM3-3B Reasoner and a HuggingFaceTB/SmolLM3-3B Weaver.

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAIN_PROCESS_PORT=29507

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

DATASET_NAME="triviaqa"

TRAIN_METHOD="sft"

MAX_PROMPT_AUG_NUM=8
MAX_INFERENCE_AUG_NUM=0
PROMPT_LATENTS_LEN=4
INFERENCE_LATENTS_LEN=4

LORA_TARGET_MODULES="[q_proj,v_proj]"

BATCH_SIZE=1

LOAD_MODEL_PATH=null


# train
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
    model.trigger.active False \
    dataset.mode ${TRAIN_METHOD} \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method ${TRAIN_METHOD} \
    run.weaver.sft.per_device_train_batch_size ${BATCH_SIZE} \
    run.weaver.sft.per_device_eval_batch_size ${BATCH_SIZE} \
    run.weaver.sft.num_train_epochs 2 \
    run.weaver.sft.learning_rate 1e-5 \
    run.weaver.sft.bf16 True \
    run.weaver.sft.gradient_accumulation_steps 1
