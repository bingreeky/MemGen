"""Scoped compatibility layer around MemGen's original multi-turn Weaver GRPO.

The grouped rewards and clipped GRPO objective remain in WeaverGRPOTrainer.
These adapters repair observation attention, chunk slicing, strict retrieval and
empty finished-row turns. They do not replace MemGen's latent insertion method.
"""
import copy
import json
from pathlib import Path
import time
import urllib.request

import torch
from trl.trainer.utils import selective_log_softmax
from transformers import TrainerCallback

from data.triviaqa.env import TriviaQAEnv
from interactions.multiturn_interaction import MultiTurnInteractionManager
from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from memgen.model.modeling_memgen import MemGenModel


def response_budget_record(token_ids, env, tokenizer, cap):
    """Inspect generated IDs BEFORE action preprocessing/chat re-templating."""
    ids = token_ids.tolist()
    eos_seen = tokenizer.eos_token_id in ids
    if eos_seen:
        ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
    else:
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    kind, _ = env._process_action(env.preprocess_action(text))
    complete = kind in ("search", "answer")
    hit_cap = len(ids) >= cap
    return {"response_tokens": len(ids), "raw_eos_seen": eos_seen,
            "response_cap": cap, "hit_response_cap": hit_cap,
            "action_type": kind, "complete_action": complete,
            "substantive_truncation": hit_cap and not eos_seen and not complete,
            "observation_tokens": 0, "observation_hit_cap": False,
            "observation_truncated": False}


def budget_audit(env, max_turns):
    turns = getattr(env, "budget_trace", [])
    return {"turns": turns, "assistant_turns": len(turns),
            "response_cap_hits": sum(t["hit_response_cap"] for t in turns),
            "substantive_response_truncations": sum(t["substantive_truncation"] for t in turns),
            "observation_truncations": sum(t["observation_truncated"] for t in turns),
            "max_turns_terminated": not env.done and len(turns) >= max_turns,
            "actual_context_tokens": getattr(env, "final_context_tokens", 0),
            "final_padded_context_tokens": getattr(env, "final_padded_context_tokens", 0),
            "peak_context_tokens": max([getattr(env, "final_context_with_latents", 0)] + [t.get("actual_context_tokens_with_latents", 0) for t in turns])}


class StrictTriviaQAEnv(TriviaQAEnv):
    """Original reward/action protocol, with strict server failures.

    The public TriviaQA environment treats malformed search actions as an
    unsuccessful search.  Preserve that behaviour locally before contacting
    the stricter batched retrieval API; actual retrieval service failures are
    still deliberately surfaced to stop the run.
    """
    def __init__(self, config):
        super().__init__(config)
        self.search_url = config["search_url"]
        self.search_timeout = config.get("search_timeout", 120)
        self.search_trace = []
        self.budget_trace = []

    @staticmethod
    def valid_search_query(query):
        """Match the retriever's documented 1..16384 character input domain."""
        return bool(query.strip()) and len(query) <= 16384

    def step(self, action):
        kind, query = self._process_action(self.preprocess_action(action))
        if kind != "search":
            return super().step(action)
        if not self.valid_search_query(query):
            # Exact public TriviaQAEnv fallback for a failed search: no trace,
            # zero reward, and permit the agent to continue on the next turn.
            self.done, self.reward = False, 0.0
            return "Cannot find corresponding pages.", self.reward, self.done
        payload = {"queries": [query], "topk": 3, "return_scores": True}
        request = urllib.request.Request(self.search_url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.search_timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or len(result.get("result", [])) != 1:
            raise RuntimeError("Invalid Search-R1 retrieval response")
        documents = result["result"][0]
        if len(documents) != 3 or any(not doc.get("document", {}).get("contents", "").strip() for doc in documents):
            raise RuntimeError("Retrieval must return three nonempty passages")
        self.search_trace.append({"query": query, "results": documents})
        self.done, self.reward = False, 0.0
        return self.explorer._passages2string(documents), self.reward, self.done


class TriviaQAGrpoInteraction(MultiTurnInteractionManager):
    """Use original multi-turn loop, retaining final masks for GRPO backprop."""
    def _latent_context_reserve(self):
        config = self.actor_rollout_wg.config
        # Original conversational forward replays each text prefix separately;
        # prior turns' latent vectors are not persisted inside the text history.
        return int(config.prompt_latents_len) + int(config.max_inference_aug_num) * int(config.inference_latents_len)

    def _build_chat_history(self, rollings):
        messages = super()._build_chat_history(rollings)
        latent_reserve = self._latent_context_reserve()
        for history, env in zip(messages, rollings["envs"]):
            ids = self.tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True)
            required = len(ids) + self.config.max_response_length + latent_reserve
            env.pending_context_budget = {"prompt_context_tokens": len(ids),
                "response_token_reserve": self.config.max_response_length,
                "latent_token_reserve": latent_reserve, "reserved_context_tokens": required,
                "context_limit": self.config.max_prompt_length}
            if required > self.config.max_prompt_length:
                raise RuntimeError(f"Context budget exceeded before generation: prompt={len(ids)} + response={self.config.max_response_length} + latents={latent_reserve} = {required} > limit={self.config.max_prompt_length}; refusing silent history truncation")
        return messages

    def _check_final_context_budget(self, output, envs):
        padded = output.batch["input_ids"].shape[1]
        latent_reserve = self._latent_context_reserve()
        actual = output.batch["attention_mask"].sum(dim=1).tolist()
        for env, count in zip(envs, actual):
            env.final_context_tokens = int(count)
            env.final_padded_context_tokens = padded
            env.final_context_with_latents = int(count) + latent_reserve
        if padded + latent_reserve > self.config.max_prompt_length:
            raise RuntimeError(f"Final training tensor exceeds context budget: padded_tokens={padded} + latents={latent_reserve} > limit={self.config.max_prompt_length}; refusing silent truncation")

    def _postprocess_responses(self, responses, envs):
        self._budget_active_envs = {id(env) for env in envs}
        for ids, env in zip(responses, envs):
            if not hasattr(env, "budget_trace"):
                env.budget_trace = []
            record = response_budget_record(ids, env, self.tokenizer, self.config.max_response_length)
            record.update(getattr(env, "pending_context_budget", {}))
            record["actual_context_tokens"] = record.get("prompt_context_tokens", 0) + record["response_tokens"]
            record["actual_context_tokens_with_latents"] = record["actual_context_tokens"] + record.get("latent_token_reserve", 0)
            record["observation_cap"] = self.config.max_obs_length
            env.budget_trace.append(record)
        return super()._postprocess_responses(responses, envs)

    def _postprocess_observations(self, observations):
        for env, text in zip(self._budget_envs, observations):
            if id(env) not in self._budget_active_envs:
                continue
            count = len(self.tokenizer.encode(text or "", add_special_tokens=False))
            env.budget_trace[-1].update(observation_tokens=count,
                observation_hit_cap=count >= self.config.max_obs_length,
                observation_truncated=count > self.config.max_obs_length)
        return super()._postprocess_observations(observations)

    def _update_interaction_history(self, rollings, responses, observations):
        histories = copy.deepcopy(rollings.no_tensor_batch["inter_histories"])
        for history, response, observation in zip(histories, responses, observations):
            if not response and not observation:
                continue  # Already finished row: do not invent supervised empty turns.
            history.append({"role": "assistant", "content": response})
            if observation:
                history.append({"role": "user", "content": observation})
        return histories

    def _build_final_outputs(self, rollings):
        output = super()._build_final_outputs(rollings)
        # Transformers 4.55.4 + this Qwen fast tokenizer returns all-zero
        # assistant_masks for a conversation starting with an assistant turn.
        # Derive the same template role spans using exact token-prefix checks.
        # No answer/observation tokens or formatting are changed.
        response_ids = output.batch["responses"]
        mask = torch.zeros_like(response_ids)
        for row, history in enumerate(rollings.no_tensor_batch["inter_histories"]):
            previous = []
            for end, message in enumerate(history, 1):
                current = self.tokenizer.apply_chat_template(history[:end], tokenize=True, add_generation_prompt=False)
                if current[:len(previous)] != previous:
                    raise RuntimeError("Chat template is not token-prefix stable; cannot derive assistant mask")
                if not torch.equal(response_ids[row, :len(current)], torch.tensor(current, dtype=response_ids.dtype)):
                    raise RuntimeError("Rendered response differs from prefix-derived assistant mask")
                if message["role"] == "assistant":
                    mask[row, len(previous):len(current)] = 1
                previous = current
            if not mask[row].any():
                raise RuntimeError("Rollout contains no assistant tokens")
        prefix_length = output.batch["prompts"].shape[1]
        output.batch["info_mask"] = torch.cat([torch.zeros_like(output.batch["prompts"]), mask], dim=1)
        return output

    def run_agent_loop(self, gen_batch):
        self._budget_envs = gen_batch.no_tensor_batch["envs"]
        self._budget_active_envs = set()
        for env in self._budget_envs:
            env.budget_trace = []
            env.final_context_tokens = env.final_padded_context_tokens = env.final_context_with_latents = 0
        was_training = self.actor_rollout_wg.training
        self.actor_rollout_wg.eval()
        try:
            # Custom generate bypasses Accelerate's forward autocast wrapper.
            # Keep FP32 trainable master parameters while running BF16 rollout.
            on_cuda = self.actor_rollout_wg.device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on_cuda):
                output = super().run_agent_loop(gen_batch)
        finally:
            self.actor_rollout_wg.train(was_training)
        self._check_final_context_budget(output, self._budget_envs)
        self.last_output = output
        self.last_envs = gen_batch.no_tensor_batch["envs"]
        return output


class TriviaQAWeaverGRPOTrainer(WeaverGRPOTrainer):
    """Compatibility fixes isolated from all existing GSM/KodCode workflows."""
    def _generate_and_score_completions(self, inputs):
        # The parent annotates dictionaries with env objects. Do not mutate sampler data.
        rows = [dict(row) for row in inputs]
        started = time.perf_counter()
        output = super()._generate_and_score_completions(rows)
        generated = self.generation_manager.last_output
        start = output["prompt_ids"].shape[1]
        output["completion_attention_mask"] = generated.batch["attention_mask"][:, start:].to(self.accelerator.device)
        batch_index = getattr(self, "rollout_batch_index", 0)
        self.rollout_batch_index = batch_index + 1
        summary = {"step": self.state.global_step, "rollout_batch": batch_index, "seconds": time.perf_counter() - started, "rows": len(rows), "assistant_template_tokens": int(output["completion_mask"].sum()), "context_tokens": int(generated.batch["attention_mask"].sum()), "search_calls": sum(len(env.search_trace) for env in self.generation_manager.last_envs)}
        budgets = [budget_audit(env, self.generation_manager.config.max_turns) for env in self.generation_manager.last_envs]
        summary["budget_audit"] = {key: sum(b[key] for b in budgets) for key in ("assistant_turns", "response_cap_hits", "substantive_response_truncations", "observation_truncations", "max_turns_terminated")}
        summary["budget_audit"]["peak_context_tokens"] = max(b["peak_context_tokens"] for b in budgets)
        summary["budget_audit"]["max_actual_context_tokens"] = max(b["actual_context_tokens"] for b in budgets)
        summary_path = Path(self.args.output_dir) / f"rollout-timing.rank{self.accelerator.process_index}.jsonl"
        with summary_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        texts = self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        rewards = self.env_class.compute_reward(texts, self.generation_manager.last_envs)
        trace = Path(self.args.output_dir) / f"rollouts.rank{self.accelerator.process_index}.jsonl"
        with trace.open("a") as handle:
            for index, (row, history, env) in enumerate(zip(rows, generated.no_tensor_batch["inter_histories"], self.generation_manager.last_envs)):
                handle.write(json.dumps({"step": self.state.global_step, "rollout_batch": batch_index, "task_id": row.get("task_id"), "messages": history, "retrievals": env.search_trace, "env_done": env.done, "reward": rewards[index], "advantage": output["advantages"][index].item(), "budget_audit": budgets[index]}, ensure_ascii=False) + "\n")
        return output

    def _get_per_token_logps(self, model, input_ids, attention_mask, labels, logits_to_keep, batch_size=None):
        # Parent _compute_loss receives assistant-only completion_mask; it must not
        # become the attention mask: search observations remain visible to the model.
        attention_mask = getattr(self, "_actual_attention_mask", None) if getattr(self, "_actual_attention_mask", None) is not None else attention_mask
        if input_ids.shape != attention_mask.shape or labels.shape != input_ids.shape:
            raise ValueError("GRPO token/attention/label shape mismatch")
        chunk = batch_size or getattr(self, "logprob_batch_size", 1)
        all_logps, all_masks = [], []
        for start in range(0, len(input_ids), chunk):
            end = start + chunk
            current_ids = input_ids[start:end]
            result = model(input_ids=current_ids, attention_mask=attention_mask[start:end], labels=labels[start:end])
            logits = result.logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature
            all_logps.append(selective_log_softmax(logits, current_ids[:, -logits_to_keep:]))
            all_masks.append((result.supervised_labels[:, -logits_to_keep:] != -100).long())
        return torch.cat(all_logps), torch.cat(all_masks)

    def _compute_loss(self, model, inputs):
        self._actual_attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_attention_mask"]], dim=1)
        try:
            return super()._compute_loss(model, inputs)
        finally:
            self._actual_attention_mask = None

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.accelerator.unwrap_model(model or self.model)
        target.load_checkpoint(resume_from_checkpoint)


class WeaverGradientAudit(TrainerCallback):
    """Log real gradients before optimizer steps; zero advantages remain honest."""
    def __init__(self, output):
        self.output = Path(output)
        self.records = []

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        sums = {"weaver": 0.0, "projection": 0.0, "reasoner": 0.0, "trigger": 0.0}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Nonfinite gradient: {name}")
            key = "projection" if "reasoner_to_weaver" in name or "weaver_to_reasoner" in name else ("weaver" if "weaver." in name else "trigger" if "trigger." in name else "reasoner")
            sums[key] += parameter.grad.detach().float().abs().sum().item()
        if sums["reasoner"] or sums["trigger"]:
            raise RuntimeError("Frozen Reasoner/Trigger received gradients")
        record = {"step": state.global_step, "gradient_abs_sums": sums}
        self.records.append(record)
        rank = int(__import__("os").environ.get("RANK", 0))
        with (self.output / f"gradients.rank{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
