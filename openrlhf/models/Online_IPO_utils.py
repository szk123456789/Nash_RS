import torch
import torch.nn as nn
from openrlhf.models import Actor
from typing import Optional, Union, Tuple

def CalculatePreference(
    reward_model: nn.Module,
    sequences1: torch.LongTensor,
    sequences2: torch.LongTensor,
    attention_mask1: Optional[torch.Tensor] = None,
    attention_mask2: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calaulate preference P(y>y')

    Args:
        reward_model: pretrained reward model
        sequences1: the first sequence y -> reward: r_1
        sequences2: the second sequence y' -> reward: r_2
        attention_mask1: the attention_mask of the first sequence
        attention_mask2: the attention_mask of the second sequence
    
    Returns:
        preference P(y>y') = Softmax(r_1, r_2)
    """

    # assume r1 shape (a1), r2 shape (a2) first
    r1 = reward_model(sequences1, attention_mask1)
    r2 = reward_model(sequences2, attention_mask2)

    exp_r1 = torch.exp(r1).view(-1, 1)  # reshape to (a1, 1)
    exp_r2 = torch.exp(r2).view(1, -1)  # reshape to (1, a2)

    result = exp_r1 / (exp_r1 + exp_r2)
    return result

def OnlineIPOReward(
    sequences,
    attention_masks,
    inputs,
    kl_coef: float,
    actor: Actor,
    reward_model: nn.Module,
    initial_model: Actor,
    actor_batch_size: int = 1,
    rejection_batch_size: int = 1,
    **generate_kwargs
) -> torch.Tensor:
    """Calculate preference P(y>pi)
    Args:
        inputs: given tokenized prompts ***(dtype: dict)***
        sequences: given responses (want to be evaluated)
        attention_masks: given attention_masks corresponding to the sequences (want to be evaluated)
        kl_coef: the coefficient of KL regularization (tau in our note)
        actor: the current policy
        reward_model: pretrained reward model used for calculating preference
        initial_model: pretrained model used for proposal in rejection sampling
        sequences1: a batch of responses from current policy -> y
        batch_size: the number of samples from rejection sampling (pi_star) to calculate the expectation (default=1)
        **generate_kwargs: configuration for generation
    
    Returns: P(y>pi) by rejection sampling
        1. generate a batch of samples from current policy pi
        2. calculate the expectation of P(y>y') by CalculatePreference
    """
    prompt_num = inputs['input_ids'].shape[0]
    prompt_length = inputs['input_ids'].shape[1]
    mask_length = inputs['attention_mask'].shape[1]

    ### generate response from actor
    inputs_actor_ids = inputs['input_ids'].repeat((1,actor_batch_size)).reshape(prompt_num * actor_batch_size, prompt_length)
    input_actor_masks = inputs['attention_mask'].repeat((1,actor_batch_size)).reshape(prompt_num * actor_batch_size, mask_length)
    inputs_actor = {
        'input_ids': inputs_actor_ids, 'attention_mask': input_actor_masks
    }
    sequence_actor, attention_mask_actor, _ = actor.generate(**inputs_actor, **generate_kwargs)
    sequence_actor = sequence_actor.reshape(prompt_num, actor_batch_size, -1)
    attention_mask_actor = attention_mask_actor.reshape(prompt_num, actor_batch_size, -1)

    preference_reward_list = []

    for prompt_index in range(prompt_num):
        input_index_ids: torch.LongTensor = inputs['input_ids'][prompt_index]
        input_index_ids = input_index_ids.reshape(1, input_index_ids.shape[0])
        mask_index_ids: torch.LongTensor = inputs['attention_mask'][prompt_index]
        mask_index_ids = mask_index_ids.reshape(1, mask_index_ids.shape[0])
        input = {
             'input_ids': input_index_ids, 'attention_mask': mask_index_ids
        }
        sequence: torch.LongTensor = sequences[prompt_index]
        sequence = sequence.reshape(1, sequence.shape[0])
        attention_mask: torch.LongTensor = attention_masks[prompt_index]
        attention_mask = attention_mask.reshape(1, attention_mask.shape[0])
        sequence_actor_prompt: torch.LongTensor = sequence_actor[prompt_index]
        attention_mask_actor_prompt: torch.LongTensor = attention_mask_actor[prompt_index]

        preference_reward = CalculatePreference(
                    reward_model=reward_model,
                    sequences1=sequence,
                    sequences2=sequence_actor_prompt,
                    attention_mask1=attention_mask,
                    attention_mask2=attention_mask_actor_prompt
                )

        
        
        preference_reward_list.append(preference_reward)
    
    reward = torch.stack(preference_reward_list)

    reward = reward.reshape(prompt_num,)

    return reward / kl_coef

def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    return log_ratio * action_mask

def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    kl_reward = -1. * kl

    r = r.clamp(min=-10, max=10)

    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward
    return reward, kl