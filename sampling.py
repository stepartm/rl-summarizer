from dataclasses import dataclass

from typing import List
import torch
import torch.nn as nn
from transformers.generation.logits_process import TopPLogitsWarper

def tensor_stats(tensor: torch.Tensor):
    return tensor.min(), tensor.max(),  tensor.mean(), tensor.std()


@dataclass
class SamplingOutput:
    sequences: torch.Tensor
    generated_sequences: torch.Tensor
    logits: List[torch.Tensor]
    attention_mask: torch.Tensor


softmax = nn.Softmax(dim=-1)
def sample(
        model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        top_p: float, temperature: float, max_new_tokens: int, eos_id: int,
        samples_per_object: int, require_no_grads: bool=True
    ) -> SamplingOutput:

    batch_size, _ = input_ids.size()

    expanded_input_ids_for_group_inference = input_ids.view(batch_size, 1, -1).repeat(1, samples_per_object, 1).view(batch_size * samples_per_object, -1)
    expanded_attention_mask_for_group_inference = attention_mask.view(batch_size, 1, -1).repeat(1, samples_per_object, 1).view(batch_size * samples_per_object, -1)

    stopping_criteria_are_met = False
    n_generated_tokens = 0
    logits_during_generation = []
    new_tokens_during_generation = []

    top_p_processor = TopPLogitsWarper(top_p=top_p)
    past_key_values = None
    while not stopping_criteria_are_met:
        with torch.inference_mode(mode=require_no_grads):
            res = model(
                input_ids=expanded_input_ids_for_group_inference if past_key_values is None else next_tokens.view(-1, 1),
                attention_mask=expanded_attention_mask_for_group_inference,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = res.past_key_values
            last_logits = res.logits[:, -1, :]
            logits_during_generation.append(last_logits)

            with torch.no_grad():
                top_p_logits = top_p_processor(
                    input_ids=None,
                    scores=last_logits.detach() / temperature
                )

            probs = softmax(top_p_logits)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            new_tokens_during_generation.append(next_tokens.view(-1, 1))

            expanded_attention_mask_for_group_inference = torch.hstack(
                (
                    expanded_attention_mask_for_group_inference,
                    torch.ones(batch_size * samples_per_object, 1).to(expanded_attention_mask_for_group_inference.device)
                )
            )

            n_generated_tokens += 1
            stopping_criteria_are_met = (next_tokens == eos_id).all() or (n_generated_tokens == max_new_tokens)

            del res

    return SamplingOutput(
        sequences=torch.cat(
            [expanded_input_ids_for_group_inference] + new_tokens_during_generation,
            dim=1
        ),
        generated_sequences=torch.cat(
            new_tokens_during_generation,
            dim=1
        ),
        logits=logits_during_generation,
        attention_mask=expanded_attention_mask_for_group_inference
    )
