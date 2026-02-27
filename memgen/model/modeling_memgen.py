import logging
import os
import random
from typing import Union

from peft import PeftModel
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig,
    DynamicCache
)
from transformers.modeling_utils import PreTrainedModel

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_utils import (
    MemGenOutputWithPast,
    MemGenLoraSwitchMixin,
    MemGenGenerationMixin,
)
from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    log_trainable_params
)

class MemGenModel(PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin):
    config_class = MemGenConfig
    INSTRUCTION_STATE = 0
    CONVERSATION_STATE = 1

    def __init__(
        self, 
        config: MemGenConfig, 
        base_tokenizer,
        reasoner_base_model: PreTrainedModel, 
        weaver_base_model: PreTrainedModel,
        trigger_base_model: PreTrainedModel,
    ):   
        super().__init__(config)
        
        self.config = config

        # insert lora adapters into weaver and trigger
        weaver_model_w_lora, trigger_model_w_lora = self._insert_lora_adapters(
            weaver_base_model, config.weaver_lora_config, trigger_base_model, config.trigger_lora_config,
        )

        # use base model with lora adapters to initiate weaver and trigger
        self.weaver = MemGenWeaver(weaver_model_w_lora, config.prompt_latents_len, config.inference_latents_len)
        self.trigger = MemGenTrigger(trigger_model_w_lora, config.trigger_active)

        # base reasoner
        self.reasoner = reasoner_base_model
        self.tokenizer = base_tokenizer
        
        # projection layers for mapping embeddings between reasoner and weaver
        reasoner_hidden_size = reasoner_base_model.config.hidden_size
        weaver_hidden_size = weaver_base_model.config.hidden_size
        self.reasoner_to_weaver = nn.Linear(reasoner_hidden_size, weaver_hidden_size)  # map reasoner input embeddings to weaver input embeddings
        self.weaver_to_reasoner = nn.Linear(weaver_hidden_size, reasoner_hidden_size)  # Map weaver hidden states to reasoner input embeddings
        
        # delimiters for detecting augmentation points
        self.delimiters: list[str] = [",", ".", "\n"]  

        self.state = None

        # postprocess
        self._postprocess_models()
        logging.info("##### MemGen Initialization #####")
        log_trainable_params(self)

    def _postprocess_models(self):
        # fix base model parameters
        fix_model_parameters(self.reasoner)  

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )

        # Normalize the tokenizer's chat template
        self.tokenizer.chat_template = CONVERSATION_TEMPLATE
    

    @property
    def device(self):
        return self.reasoner.device
    
    def _forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> torch.Tensor:
        # preprocess inputs
        assert input_ids.shape == attention_mask.shape == labels.shape
        
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.config.max_inference_aug_num  # Limit the number of inference augmentation points to avoid excessive augmentation
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size

        # select augment idx
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )
        
        # origin inputs embeds
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
                
        # Initialize the start index and empty tensors for accumulating processed segments
        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        # Iterate over the selected augmentation points
        for aug_point_idx in augmentation_indices:
            # Slice the current segment of original embeddings and attention mask
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            # Concatenate the current segment to the accumulated embeddings and masks
            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            # Map reasoner embeddings to weaver embeddings for augmentation
            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)

            # Determine whether this point is the end of the prompt (prompt augmentation)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx-1] == -100).all().item()
            
            # Depending on type, use weaver to augment prompt or inference
            if is_prompt_end_aug:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                ) 

            # Map weaver hidden states back to reasoner embeddings
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

            # Update accumulated embeddings and masks with the newly augmented segment
            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx
            
            # Update latent mask for the newly added latent embeddings
            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)
            
        # Process the remaining segment after the last augmentation point
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool)
        
        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits
        
        # Identify valid positions in logits (positions that should contribute to loss)
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted
        
        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))  
        # assert shifted.sum() == current_latents_mask.sum()
        # assert valid_logits.shape[:2] == input_ids.shape
        return valid_logits
    
    def _instructional_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for single-turn instructional data (no multi-turn conversation required).

        This method is used for instruction-following tasks (SFT), where the input
        consists of a single instruction and the corresponding labels. It directly
        delegates to the single-turn forward method `_forward`.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing input token IDs.
            attention_mask (torch.Tensor): Tensor indicating padding positions.
            labels (torch.Tensor): Tensor containing the target labels for supervised fine-tuning.
            **kwargs: Additional keyword arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - logits: The output logits from the model for each input token.
                - labels: The same as input labels, used for loss computation.
        """
        # raise RuntimeError()
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        # For Instruction SFT, labels remain the same as input
        return logits, labels

    def _conversational_forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,   
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Forward pass for conversational (multi-turn) data.

        Multi-turn forward is constructed by sequentially calling the single-turn forward
        for each conversation turn. Latents inserted in turn i-1 are not visible to turn i.

        Args:
            input_ids (torch.Tensor): Input token IDs, shape (1, seq_len). Batch size must be 1.
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor): Target labels for supervised fine-tuning (-100 for ignore positions).
            **kwargs: Additional arguments passed to `_forward`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - all_logits: Logits for the entire sequence, with zeros for unsupervised positions.
                - all_labels: Labels for the entire sequence, with -100 for unsupervised positions.
        """
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        # Identify single-turn segments within the conversation based on labels
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        # Compute the start and end indices of valid supervised segments
        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()  # Transition 0 -> 1
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()          # Transition 1 -> 0
        if len(ends) < len(valid_starts):
            ends.append(seq_len)  # 自动补充最后一个 token 的 (index + 1) 作为最后一个序列的末尾
        assert len(valid_starts) == len(ends)
        
        # Build triplets (start of previous segment, start of supervised segment, end of supervised segment)
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e
        
        # If there are more segments than allowed, randomly select self.max_prompt_aug_num segments
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]

        # Initialize tensors to store logits and labels for the entire sequence
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        # Loop over each conversation turn and perform single-turn forward if supervised
        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                # cur_labels only used for _forward, does not represent the true supervision range
                # cur_labels = labels[0, :end].clone().unsqueeze(0)
                # cur_labels[0, :valid_start] = -100  # Mask tokens before supervision start
                cur_labels = torch.full((1, end), -100, device=device)
                cur_labels[0, valid_start:end] = labels[0, valid_start:end]

                # Single-turn forward for the current conversation segment
                logits = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)
                
                # Update overall logits and labels with the results of this segment
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        # Return logits and labels:
        # - supervised positions retain computed logits and original labels
        # - unsupervised positions have logits = 0 and labels = -100
        return all_logits, all_labels

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:  
        tokenizer = self.tokenizer

        # Ensure labels are provided, required for training the reasoning processor
        assert labels is not None, "Reasoning Processor requires input labels for training"
        
        # Determine whether the input is single-turn (instruction) or multi-turn (conversation)
        labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer)
       
        # Use only the first data sample of each dataset to determine the model state
        if self.state is None:  
            self.state = MemGenModel.CONVERSATION_STATE if self._is_conversation(input_ids, tokenizer) else MemGenModel.INSTRUCTION_STATE

        if self.state == MemGenModel.INSTRUCTION_STATE:
            forward_func = self._instructional_forward
        elif self.state == MemGenModel.CONVERSATION_STATE:
            forward_func = self._conversational_forward
        else:
            raise RuntimeError(f"Unexpected model state: {self.state}")
        
        batch_size = 1  # Currently process one sequence per batch
        iter_num = input_ids.size(0) // batch_size

        # Forward pass per batch
        logits, supervised_labels = [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            # Call the appropriate forward function (instruction or conversation)
            batch_logits, batch_supervised_labels = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
        
        # Concatenate results from all batches
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        # Compute causal language modeling loss (shifted by one)
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        # assert shift_logits.shape[:-1] == shift_labels.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Return model outputs
        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels  # Positions in input_ids that are supervised
        return outputs

    # @torch.no_grad()
    # def generate(
    #     self, 
    #     input_ids: torch.Tensor, 
    #     attention_mask: torch.Tensor,
    #     generation_config: GenerationConfig = None, 
    #     return_augmentation_mask: bool = False,
    #     **kwargs
    # ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]: 
        
    #     tokenizer = self.tokenizer
    #     reasoner = self.reasoner
    #     weaver = self.weaver
    #     max_augment_num = self.config.max_inference_aug_num
    #     invalid_token_id = -100

    #     # preproecess inputs
    #     input_ids = input_ids.to(self.device)
    #     attention_mask = attention_mask.to(self.device)
    #     max_new_tokens = generation_config.max_new_tokens
    #     pad_token_id = tokenizer.pad_token_id
    #     eos_token_id = tokenizer.eos_token_id
    #     prompt_len = input_ids.size(1)

    #     inputs_embeds = reasoner.get_input_embeddings()(input_ids)
    #     B, _, hidden_size = inputs_embeds.shape
    #     device = inputs_embeds.device
        
    #     # --- generation loop ---
    #     current_inputs_embeds = inputs_embeds
    #     current_attention_mask = attention_mask
    #     current_position_ids = self._generate_position_ids(current_attention_mask)
    #     current_input_ids = input_ids
    #     current_cache: DynamicCache = None

    #     # Generation Loop Initialization
    #     sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        
    #     # NOTE - Whether to call the trigger and insert latent memory before generating the token at this position
    #     # - augmentation_pos[b][i] == -100: For the b-th sequence, no augmentation was sampled before generating the i-th token
    #     # - augmentation_pos[b][i] == 0: For the b-th sequence, augmentation was sampled before generating the i-th token, but the trigger decided NOT to insert latent memory
    #     # - augmentation_pos[b][i] == 1: For the b-th sequence, augmentation was sampled before generating the i-th token, and the trigger decided to insert latent memory
    #     augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device) 

    #     generation_config = GenerationConfig(
    #         do_sample=False,
    #         pad_token_id=pad_token_id,
    #         eos_token_id=eos_token_id,
    #         use_cache=False,
    #         max_new_tokens=max_new_tokens
    #     )
    #     # Perform generation for the remaining tokens using the reasoner
    #     generated = reasoner.generate(
    #         inputs_embeds=current_inputs_embeds,
    #         attention_mask=current_attention_mask,
    #         generation_config=generation_config
    #     )
    #     current_input_ids = torch.cat([current_input_ids, generated], dim=1)

    #     # postprocess
    #     new_generated_len = current_input_ids.size(1) - prompt_len
    #     augmentation_pos = augmentation_pos[:, :new_generated_len]
        
    #     self._check_generate(
    #         current_input_ids[:, prompt_len:],
    #         augmentation_pos
    #     )
        
    #     if return_augmentation_mask:
    #         return (current_input_ids, augmentation_pos)
    #     else:
    #         return current_input_ids

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None, 
        return_augmentation_mask: bool = False,
        **kwargs
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]: 
        
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        max_augment_num = self.config.max_inference_aug_num
        invalid_token_id = -100

        # preproecess inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        # --- generation loop ---
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_cache: DynamicCache = None

        # Generation Loop Initialization
        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        
        # NOTE - Whether to call the trigger and insert latent memory before generating the token at this position
        # - augmentation_pos[b][i] == -100: For the b-th sequence, no augmentation was sampled before generating the i-th token
        # - augmentation_pos[b][i] == 0: For the b-th sequence, augmentation was sampled before generating the i-th token, but the trigger decided NOT to insert latent memory
        # - augmentation_pos[b][i] == 1: For the b-th sequence, augmentation was sampled before generating the i-th token, and the trigger decided to insert latent memory
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device) 

        for i in range(max_new_tokens):
            
            assert current_inputs_embeds.shape[:2] == current_attention_mask.shape == current_position_ids.shape
            augment_decision = self._should_augment(
                current_input_ids, 
                sentence_augment_count=sentence_augment_count, 
                do_sample=generation_config.trigger_do_sample,
                temperature=generation_config.temperature,
                is_prompt=(i==0)  
            )
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            # If there are sentences to augment, apply augmentation; others remain with left padding
            if len(augment_indices) > 0:
                # Increment the augmentation count for sentences that are being augmented
                if i != 0:  
                    sentence_augment_count[augment_indices] += 1

                # Select embeddings, attention masks, and position IDs for sentences to be augmented
                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]
                candidate_position_ids = current_position_ids[augment_indices]
                
                # Perform inference augmentation using the weaver
                weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
                if i == 0:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )                    
                else:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )
                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
                
                candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
                candidate_attention_mask = torch.cat([candidate_attention_mask, attn_mask], dim=1)
                
                # Create a single merged tensor for all sequences
                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)
                   
                # Directly place augmented and non-augmented sequences
                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask
                
                # Non-augmented sequences now include both -100 and 0
                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    # dynamic left padding
                    non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                    non_aug_attention_mask = current_attention_mask[non_augment_indices]
                    pad_len = weaver.prompt_latents_num if i == 0 else weaver.inference_latents_num
                    non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                        non_aug_inputs_embeds, non_aug_attention_mask, None, pad_len
                    )
                    
                    merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                    merged_attention_mask[non_augment_indices] = non_aug_attention_mask
                
                current_inputs_embeds = merged_inputs_embeds
                current_attention_mask = merged_attention_mask
                current_position_ids = self._generate_position_ids(current_attention_mask)
                current_cache = None  

            # Check if all sequences have reached the maximum number of augmentations
            if (sentence_augment_count >= max_augment_num).all():
                # Adjust the remaining generation length
                generation_config_continue = GenerationConfig(
                    do_sample=generation_config.weaver_do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=False,
                    max_new_tokens=max_new_tokens-i
                )
                # Perform generation for the remaining tokens using the reasoner
                generated = reasoner.generate(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    generation_config=generation_config_continue
                )
                current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                break            

            if current_cache is not None:
                assert current_inputs_embeds.size(1) == current_cache.get_seq_length() + 1
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[:, -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids

            outputs = reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=current_cache
            )
            current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = self._append_one_step(
                outputs, 
                current_inputs_embeds, 
                current_attention_mask, 
                current_position_ids, 
                current_input_ids, 
                do_sample=generation_config.weaver_do_sample, 
                temperature=generation_config.temperature
            )
            current_cache = outputs.past_key_values

            # If all sequences in the batch have already generated an EOS token, stop early
            if (current_input_ids[:, -1] == eos_token_id).all():
                break  

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        
        # postprocess
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        
        self._check_generate(
            current_input_ids[:, prompt_len:],
            augmentation_pos
        )
        
        if return_augmentation_mask:
            return (current_input_ids, augmentation_pos)
        else:
            return current_input_ids
    
    @classmethod
    def from_config(cls, config_dict: dict):
        # base LLM
        model_name = config_dict.get("model_name")

        # max augment numbers
        max_prompt_aug_num = config_dict.get("max_prompt_aug_num", 1)
        max_inference_aug_num = config_dict.get("max_inference_aug_num", 5)

        # weaver configs
        weaver_config = config_dict.get("weaver", {})
        prompt_latents_len = weaver_config.get("prompt_latents_len", 8)
        inference_latents_len = weaver_config.get("inference_latents_len", 8)
        weaver_lora_config_dict = weaver_config.get("lora_config", None)
        weaver_model_name = weaver_config.get("model_name", None)

        # trigger configs
        trigger_config = config_dict.get("trigger", {})
        trigger_active = trigger_config.get("active", False)
        trigger_lora_config_dict = trigger_config.get("lora_config", None)
        trigger_model_name = trigger_config.get("model_name", None)

        # build MemGenConfig
        from transformers import AutoConfig
        memgen_config = AutoConfig.from_pretrained(model_name)
        memgen_config = MemGenConfig.from_pretrained(
            model_name, 
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_lora_config=weaver_lora_config_dict,
            trigger_active=trigger_active,
            trigger_lora_config=trigger_lora_config_dict
        )
        
        # load pretrained base models
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        reasoner_base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        weaver_base_model = AutoModelForCausalLM.from_pretrained(weaver_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        trigger_base_model = AutoModelForCausalLM.from_pretrained(trigger_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        
        # instantiate MemGen Model
        load_model_path = config_dict.get("load_model_path", None)
        
        if not load_model_path:
            model = cls(
                config=memgen_config, 
                base_tokenizer=base_tokenizer,
                reasoner_base_model=reasoner_base_model, 
                weaver_base_model=weaver_base_model,
                trigger_base_model=trigger_base_model
            )
        else:
            model = cls.from_pretrained(
                load_model_path, 
                config=memgen_config,
                base_tokenizer=base_tokenizer,
                reasoner_base_model=reasoner_base_model,
                weaver_base_model=weaver_base_model,
                trigger_base_model=trigger_base_model
            )
        
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        self.config.save_pretrained(save_directory)

        torch.save(
            {
                "reasoner_to_weaver": self.reasoner_to_weaver.state_dict(),
                "weaver_to_reasoner": self.weaver_to_reasoner.state_dict(),
            },
            os.path.join(save_directory, "projs.bin"),
        )

        torch.save(
            {
                "prompt_query_latents": self.weaver.prompt_query_latents.data,
                "inference_query_latents": self.weaver.inference_query_latents.data,
                "prompt_latent_ln": self.weaver.prompt_latent_ln.state_dict(),
                "inference_latent_ln": self.weaver.inference_latent_ln.state_dict(),
                "prompt_latent_scale": self.weaver.prompt_latent_scale.data,
                "inference_latent_scale": self.weaver.inference_latent_scale.data,
            },
            os.path.join(save_directory, "weaver.bin"),
        )

        torch.save(
            {
                "output_layer": self.trigger.output_layer.state_dict(),
            },
            os.path.join(save_directory, "trigger.bin"),
        )

        self.weaver.model.save_pretrained(os.path.join(save_directory, "weaver"))
        self.trigger.model.save_pretrained(os.path.join(save_directory, "trigger"))


    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        *,
        config,
        base_tokenizer,
        reasoner_base_model,
        weaver_base_model,
        trigger_base_model,
    ):
        model = cls(
            config=config,
            base_tokenizer=base_tokenizer,
            reasoner_base_model=reasoner_base_model,
            weaver_base_model=weaver_base_model,
            trigger_base_model=trigger_base_model,
        )

        proj_path = os.path.join(load_directory, "projs.bin")
        proj_state = torch.load(proj_path, map_location="cpu")
        model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
        model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])

        weaver_path = os.path.join(load_directory, "weaver.bin")
        weaver_state = torch.load(weaver_path, map_location="cpu")
        model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
        model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
        model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
        model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
        model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
        model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])

        trigger_path = os.path.join(load_directory, "trigger.bin")
        trigger_state = torch.load(trigger_path, map_location="cpu")
        model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])

        model.weaver.model = PeftModel.from_pretrained(
            model.weaver.model.base_model,
            os.path.join(load_directory, "weaver", "weaver"),
            adapter_name=MemGenWeaver.adapter_name,
        )
        model.weaver.model.set_adapter(MemGenWeaver.adapter_name)

        model.trigger.model = PeftModel.from_pretrained(
            model.trigger.model.base_model,
            os.path.join(load_directory, "trigger", "trigger"),
            adapter_name=MemGenTrigger.adapter_name,
        )
        model.trigger.model.set_adapter(MemGenTrigger.adapter_name)

        logging.info("##### MemGen from Pretrained #####")
        log_trainable_params(model)

        return model

