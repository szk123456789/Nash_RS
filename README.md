# Statistical Impossibility and Possibility of Aligning LLMs with Human Preferences: From Condorcet Paradox to Nash Equilibrium

This repository is the official PyTorch implementation of paper: Statistical Impossibility and Possibility of Aligning LLMs with Human Preferences: From Condorcet Paradox to Nash Equilibrium.

**Kaizhao Liu, Qi Long, Zhekun Shi, Weijie J. Su, Jiancong Xiao (using alphabetical order)**

**arXiv**: [https://arxiv.org/abs/2503.10990](https://arxiv.org/abs/2503.10990)

## Abstract

Aligning large language models (LLMs) with diverse human preferences is critical for ensuring
fairness and informed outcomes when deploying these models for decision-making. In this
paper, we seek to uncover fundamental statistical limits concerning aligning LLMs with human
preferences, with a focus on the probabilistic representation of human preferences and the
preservation of diverse preferences in aligned LLMs. We first show that human preferences
can be represented by a reward model if and only if the preference among LLM-generated
responses is free of any Condorcet cycle. Moreover, we prove that Condorcet cycles exist
with probability converging to one exponentially fast under a probabilistic preference model,
thereby demonstrating the *impossibility* of fully aligning human preferences using reward-based
approaches such as reinforcement learning from human feedback. Next, we explore the conditions
under which LLMs would employ mixed strategies—meaning they do not collapse to a single
response—when aligned in the limit using a non-reward-based approach, such as Nash learning
from human feedback (NLHF). We identify a necessary and sufficient condition for mixed
strategies: the absence of a response that is preferred over all others by a majority. As a blessing,
we prove that this condition holds with high probability under the probabilistic preference model,
thereby highlighting the statistical *possibility* of preserving minority preferences without explicit
regularization in aligning LLMs. Finally, we leverage insights from our statistical results to
design a novel, computationally efficient algorithm for finding Nash equilibria in aligning LLMs
with NLHF. Our experiments show that Llama-3.2-1B, aligned with our algorithm, achieves a
win rate of 60.55% against the base model.

## Main Results

## Code

Throughout the experimental process, we implement our algorithm (Nash-RS) and baseline method (Online-IPO) using [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We directly use PPO implemented in [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for the comparison. 

To run PPO method, one can use the following command

```shell
deepspeed --module openrlhf.cli.train_ppo \
  --pretrain /root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct \
  --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
  --save_path /root/autodl-tmp/ckpt/models_PPO_512prompt_trivial-1th \
  --ckpt_path /root/autodl-tmp/ckpt/checkpoints_PPO_512prompt_trivial-1th \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 4 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --actor_init_on_gpu \
  --truncate_steps 512
```

To run Nash-RS (ours) and Online-IPO, one can use `nash_main.sh` and `ipo_main.sh` respectively.

## Citation

