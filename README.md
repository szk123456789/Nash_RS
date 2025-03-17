# What We Have Changed in OpenRLHF for Nash-RS

- add `openrlhf.models.NashRS_utils.py`

- add `openrlhf.trainer.ppo_utils.NashRS_experience_maker.py`, also add this to `openrlhf.trainer.ppo_utils.__init__.py`

- add `openrlhf.trainer.NashRS_ppo_trainer.py`, also add this to `openrlhf.trainer.__init__.py`

- add `nash_main.py`, `nash.main.sh`

  Similar things have been done to implement Online-IPO.

# Running on 4$\times$A800-80GB

## PPO Command

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
