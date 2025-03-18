set -x 

read -r -d '' training_commands <<EOF
/root/ipo_main.py \
   --pretrain $1 \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path $2 \
   --ckpt_path $3 \
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
   --actor_learning_rate 5e-6 \
   --critic_learning_rate 9e-5 \
   --init_kl_coef 0.15 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --actor_init_on_gpu \
   --actor_batch_size 1 \
   --rejection_batch_size 1 \
   --truncate_steps 512
EOF
    # --wandb [WANDB_TOKENS] or True (use wandb login command)

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
