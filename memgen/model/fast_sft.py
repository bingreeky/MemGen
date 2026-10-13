"""Event-batched single-turn Weaver SFT (opt-in via `model.fast_sft`).

This is teacher forcing, not autoregressive generation. Weaver events within each
example stay sequential, while independent examples share every model call, instead
of the default path's one-example-at-a-time loop. The loss and gradients match the
default path up to floating-point summation order.
"""
import torch
from torch.nn import functional as F


def event_batched_forward(model, input_ids, attention_mask, labels):
    """Return reasoner logits aligned with `labels`, as the default per-example path does."""
    if model.config.trigger_active:
        raise ValueError("Event-batched Weaver SFT expects an inactive Trigger")
    if model.config.max_prompt_aug_num != 1:
        raise ValueError("Event-batched Weaver SFT supports exactly one prompt insertion")
    boundaries = (labels[:, 1:] != -100) & (labels[:, :-1] == -100)
    if not torch.all(boundaries.sum(1) == 1):
        raise ValueError("Event-batched Weaver SFT needs exactly one prompt boundary per row")

    # Insertion points: the prompt boundary, then up to max_inference_aug_num supervised delimiters.
    delimiter_ids = torch.tensor(sorted(model._get_delimiter_token_ids(model.tokenizer, model.delimiters)),
                                 device=input_ids.device, dtype=input_ids.dtype)
    candidates = ((labels[:, 1:] != -100) & (labels[:, :-1] != -100)
                  & torch.isin(input_ids[:, :-1], delimiter_ids))
    candidates &= candidates.long().cumsum(1) <= model.config.max_inference_aug_num
    points = [[] for _ in range(input_ids.shape[0])]
    for row, index in (boundaries | candidates).nonzero().cpu().tolist():   # one host transfer per batch
        points[row].append(index + 1)

    original = model.reasoner.get_input_embeddings()(input_ids)
    batch, length, width = original.shape
    histories = [original[row, :points[row][0]] for row in range(batch)]
    masks = [attention_mask[row, :points[row][0]] for row in range(batch)]
    logit_indices = torch.arange(length, device=input_ids.device)[None].expand(batch, -1).clone()
    original_positions = torch.arange(length, device=input_ids.device)
    for event in range(max(map(len, points))):
        active = [row for row in range(batch) if event < len(points[row])]
        longest = max(histories[row].shape[0] for row in active)
        prefixes = torch.stack([F.pad(histories[row], (0, 0, longest - histories[row].shape[0], 0)) for row in active])
        prefix_masks = torch.stack([F.pad(masks[row], (longest - masks[row].shape[0], 0)) for row in active])
        augment = model.weaver.augment_prompt if event == 0 else model.weaver.augment_inference
        hidden, latent_mask, _ = augment(model.reasoner_to_weaver(prefixes), prefix_masks,
                                         model._generate_position_ids(prefix_masks))
        latents = model.weaver_to_reasoner(hidden)
        latent_length = latents.shape[1]
        for local, row in enumerate(active):
            point = points[row][event]
            next_point = points[row][event + 1] if event + 1 < len(points[row]) else length
            histories[row] = torch.cat((histories[row], latents[local], original[row, point:next_point]), dim=0)
            masks[row] = torch.cat((masks[row], latent_mask[local], attention_mask[row, point:next_point]), dim=0)
            # The last inserted latent, not the preceding text token, predicts the next supervised token.
            logit_indices[row] += latent_length * (original_positions >= point - 1)

    longest = max(x.shape[0] for x in histories)
    extended = torch.stack([F.pad(x, (0, 0, 0, longest - x.shape[0])) for x in histories])
    extended_mask = torch.stack([F.pad(x, (0, longest - x.shape[0])) for x in masks])
    position_ids = model._generate_position_ids(extended_mask)

    logits = model.reasoner(inputs_embeds=extended, attention_mask=extended_mask, position_ids=position_ids).logits
    return logits.gather(1, logit_indices[..., None].expand(-1, -1, logits.shape[-1]))
