#!/bin/bash
# Warm-start Weaver GRPO from the matching SFT or an earlier GRPO checkpoint.
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

REASONER_MODEL="${REASONER_MODEL:-HuggingFaceTB/SmolLM3-3B}"
WEAVER_MODEL="${WEAVER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRIGGER_MODEL="${TRIGGER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-hf_upload/SmolLM3-3B/triviaqa/weaver-qwen2.5-1.5b/weaver-sft/pn=8_pl=4_in=0_il=4/model}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8001/retrieve}"
LORA_TARGET_MODULES="[q_proj,v_proj]"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    main.py \
    --cfg-path configs/latent_memory/triviaqa.yaml \
    --options \
    model.model_name "${REASONER_MODEL}" \
    model.weaver.model_name "${WEAVER_MODEL}" \
    model.trigger.model_name "${TRIGGER_MODEL}" \
    model.load_model_path "${LOAD_MODEL_PATH}" \
    model.attn_implementation sdpa \
    model.max_prompt_aug_num 8 \
    model.max_inference_aug_num 0 \
    model.weaver.prompt_latents_len 4 \
    model.weaver.inference_latents_len 4 \
    model.weaver.lora_config.target_modules "${LORA_TARGET_MODULES}" \
    model.trigger.lora_config.target_modules "${LORA_TARGET_MODULES}" \
    model.trigger.active False \
    dataset.mode grpo \
    dataset.search_url "${SEARCH_URL}" \
    dataset.search_timeout 600 \
    dataset.grpo.train_size 2000 \
    dataset.grpo.seed 42 \
    run.mode train \
    run.train_weaver True \
    run.train_trigger False \
    run.train_weaver_method grpo \
    run.weaver.grpo.per_device_train_batch_size 1 \
    run.weaver.grpo.per_device_eval_batch_size 1 \
    run.weaver.grpo.gradient_accumulation_steps 8 \
    run.weaver.grpo.steps_per_generation 8 \
    run.weaver.grpo.num_generations 8 \
    run.weaver.grpo.num_iterations 1 \
    run.weaver.grpo.num_train_epochs 1 \
    run.weaver.grpo.learning_rate 1e-5 \
    run.weaver.grpo.temperature 1.0 \
    run.weaver.grpo.beta 0.0 \
    run.weaver.grpo.max_prompt_length 1024 \
    run.weaver.grpo.max_completion_length 2048 \
    run.weaver.grpo.mask_truncated_completions False \
    run.weaver.grpo.bf16 True \
    run.weaver.grpo.disable_dropout True \
    run.weaver.grpo.use_vllm False \
    run.weaver.grpo.report_to none \
    run.weaver.grpo.eval_strategy '"no"' \
    run.weaver.grpo.load_best_model_at_end False \
    run.weaver.grpo.save_strategy steps \
    run.weaver.grpo.save_steps 100 \
    run.weaver.grpo.save_total_limit 6 \
    run.weaver.grpo.logging_steps 1 \
    run.interaction.max_turns 5 \
    run.interaction.max_start_length 1024 \
    run.interaction.max_prompt_length 16384 \
    run.interaction.max_response_length 2048 \
    run.interaction.max_obs_length 512 \
    run.interaction.temperature 1.0 \
    "$@"
