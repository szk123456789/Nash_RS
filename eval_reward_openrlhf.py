"""This code is adopted from openrlhf.cli.train_ppo.py and openrlhf.trainer.ppo_trainer.py
"""
import argparse
import json
from datetime import datetime
import torch
import os
import numpy as np
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.datasets import PromptDataset
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

def main(args):
    
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())
    
    reward_model = get_llm_for_sequence_regression(
        args.reward_pretrain,
        "reward",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
    )
    get_tokenizer(args.reward_pretrain, reward_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    actor.eval()
    reward_model.eval()
    strategy.print(actor)
    strategy.print(reward_model)

    (
        actor,
        reward_model,
    ) = strategy.prepare(
        actor,
        reward_model,
        is_rlhf=True,
    )

    # tokenizer
    def tokenize_fn(texts, max_length, device):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}
    
    # Initialize the output list
    eval_results_list_train = []
    eval_results_list_test = []
    reward_mean_train: float = 0.
    reward_mean_test: float = 0.
    sequences_dict_train = {}
    sequences_dict_test = {}
    attention_mask_dict_train = {}
    attention_mask_dict_test = {}
    reward_dict_train = {}
    reward_dict_test = {}
    
    # prepare prompt dataset
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    
    # prepare dataloader
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
    steps = 1
    eval_train_steps = min(args.training_steps, args.eval_train_steps)
    eval_test_steps = args.eval_test_steps
    assert eval_test_steps + args.training_steps < prompts_dataloader.__len__()

    for rand_prompts in prompts_dataloader:

        # evaluation on trained prompts
        if steps <= eval_train_steps:
            strategy.print(f"[{steps}/{eval_train_steps}] evaluate (training)" + "-" * 50)

            inputs = tokenize_fn(rand_prompts, args.prompt_max_len, device="cuda")
            with torch.no_grad():
                sequences, attention_mask, _ = actor.generate(**inputs,
                    do_sample=True,
                    max_new_tokens=args.generate_max_len,
                    max_length=args.max_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                reward: torch.Tensor = reward_model(sequences, attention_mask)
            strategy.print(f"sequences shape: {sequences.shape}")
            strategy.print(f"attention_mask shape: {attention_mask.shape}")
            strategy.print(f"reward: {reward}")
            reward_mean_train += reward.mean().item()

            sequences_dict_train.update({f"{steps}": sequences})
            attention_mask_dict_train.update({f"{steps}": attention_mask})
            reward_dict_train.update({f"{steps}": reward})

            output = tokenizer.batch_decode(sequences, skip_special_tokens=True)

            new_example = {
                "output": output,
                "reward": [reward[index].item() for index in range(reward.shape[0])]
            }
        
            # Append to the output list
            eval_results_list_train.append(new_example)

            steps += 1
    
        elif steps > args.training_steps and steps <= eval_test_steps + args.training_steps:
            strategy.print(f"[{steps}/{eval_test_steps + args.training_steps}] evaluate (test)" + "-" * 50)

            inputs = tokenize_fn(rand_prompts, args.prompt_max_len, device="cuda")
            with torch.no_grad():
                sequences, attention_mask, _ = actor.generate(**inputs,
                    do_sample=True,
                    max_new_tokens=args.generate_max_len,
                    max_length=args.max_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                reward: torch.Tensor = reward_model(sequences, attention_mask)
            strategy.print(f"sequences shape: {sequences.shape}")
            strategy.print(f"attention_mask shape: {attention_mask.shape}")
            strategy.print(f"reward: {reward}")
            reward_mean_test += reward.mean().item()

            sequences_dict_test.update({f"{steps}": sequences})
            attention_mask_dict_test.update({f"{steps}": attention_mask})
            reward_dict_test.update({f"{steps}": reward})

            output = tokenizer.batch_decode(sequences, skip_special_tokens=True)

            new_example = {
                "output": output,
                "reward": [reward[index].item() for index in range(reward.shape[0])]
            }
        
            # Append to the output list
            eval_results_list_test.append(new_example)

            steps += 1
        
        else:

            steps += 1
    
    reward_mean_train = reward_mean_train / eval_train_steps
    reward_mean_test = reward_mean_test / eval_test_steps

    eval_results_list_train.append({"mean reward": reward_mean_train})
    eval_results_list_test.append({"mean reward": reward_mean_test})
    
    os.makedirs(name=args.output_path, exist_ok=True)
    np.save(f"{args.output_path}/openrlhf_collection_sequences_train.npy", sequences_dict_train, allow_pickle=True)
    np.save(f"{args.output_path}/openrlhf_collection_sequences_test.npy", sequences_dict_test, allow_pickle=True)
    np.save(f"{args.output_path}/openrlhf_collection_attention_mask_train.npy", attention_mask_dict_train, allow_pickle=True)
    np.save(f"{args.output_path}/openrlhf_collection_attention_mask_test.npy", attention_mask_dict_test, allow_pickle=True)
    np.save(f"{args.output_path}/openrlhf_collection_reward_train.npy", reward_dict_train, allow_pickle=True)
    np.save(f"{args.output_path}/openrlhf_collection_reward_test.npy", reward_dict_test, allow_pickle=True)

    output_json_path = f"{args.output_path}/eval_results_train.json"

    with open(output_json_path, mode="w") as json_file:
        json.dump(eval_results_list_train, json_file, indent=2)  # Ensures [] and commas are properly formatted

    output_json_path = f"{args.output_path}/eval_results_test.json"

    with open(output_json_path, mode="w") as json_file:
        json.dump(eval_results_list_test, json_file, indent=2)  # Ensures [] and commas are properly formatted
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--training_steps", type=int, default=512)
    parser.add_argument("--eval_train_steps", type=int, default=256)
    parser.add_argument("--eval_test_steps", type=int, default=256)


    # Checkpoint
    parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/ckpt/model_PPO")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="/root/autodl-tmp/ckpt/checkpoints_PPO")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--truncate_steps", type=int, default=625, help="Truncated steps for training")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None
    
    main(args)