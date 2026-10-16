#!/bin/bash
# Two stages: (1) Weaver SFT, (2) Trigger GRPO on top of the frozen Weaver from stage 1.

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507

# Count the visible GPUs automatically.
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1


REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

DATASET_NAME="kodcode"

MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=4
INFERENCE_LATENTS_LEN=4

LORA_TARGET_MODULES="[q_proj,v_proj]"

MEMGEN_OPTIONS="model.model_name ${REASONER_MODEL} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.weaver.lora_config.target_modules ${LORA_TARGET_MODULES} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.lora_config.target_modules ${LORA_TARGET_MODULES}"

# main.py writes each run to .cache/train/<dataset>/<reasoner>/pn=.._pl=.._in=.._il=.._<time>/model
RUN_GLOB=".cache/train/${DATASET_NAME}/${REASONER_MODEL#*/}/pn=${MAX_PROMPT_AUG_NUM}_pl=${PROMPT_LATENTS_LEN}_in=${MAX_INFERENCE_AUG_NUM}_il=${INFERENCE_LATENTS_LEN}_*"


# Stage 1: Weaver SFT (Trigger inactive)
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=${NUM_GPUS} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    ${MEMGEN_OPTIONS} \
    model.load_model_path null \
    model.trigger.active False \
    datasets.mode sft \
    run.mode train \
    run.train_weaver True \
    run.train_weaver_method sft \
    run.train_trigger False \
    run.weaver.sft.per_device_train_batch_size 1 \
    run.weaver.sft.gradient_accumulation_steps 1 \
    run.weaver.sft.learning_rate 1e-5 \
    run.weaver.sft.num_train_epochs 2 \
    run.weaver.sft.bf16 True || exit 1

WEAVER_PATH=$(ls -td ${RUN_GLOB}/model | head -1)
echo "Weaver checkpoint: ${WEAVER_PATH}"


# Stage 2: Trigger GRPO (Weaver frozen)
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    --num_processes=${NUM_GPUS} \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    ${MEMGEN_OPTIONS} \
    model.load_model_path ${WEAVER_PATH} \
    model.trigger.active True \
    datasets.mode grpo \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method grpo \
    run.trigger.grpo.per_device_train_batch_size 8 \
    run.trigger.grpo.per_device_eval_batch_size 8 \
    run.trigger.grpo.num_generations 8 \
    run.trigger.grpo.gradient_accumulation_steps 1 \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.bf16 True
