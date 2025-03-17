#!/bin/bash
bash /root/eval_reward_openrlhf.sh /root/.cache/huggingface/hub/models_OnlineIPO_512prompt_trivial-3th /root/autodl-tmp/ckpt/Eval_openrlhf_Llama-3.2-1B-Instruct_OnlineIPO_3 512 128 128
bash /root/eval_reward_openrlhf.sh /root/.cache/huggingface/hub/models_OnlineIPO_512prompt_trivial-4th /root/autodl-tmp/ckpt/Eval_openrlhf_Llama-3.2-1B-Instruct_OnlineIPO_4 512 128 128