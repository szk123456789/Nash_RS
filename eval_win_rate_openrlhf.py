import numpy as np
import torch
import logging
from prettytable import PrettyTable

def get_logger(name, dir):
    logger = logging.getLogger(f'{name}')
    logger.setLevel(level=logging.DEBUG)
    filehandler = logging.FileHandler(f'{dir}/{name}.log')
    filehandler.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    return logger

def show(value: float) -> float:
    return float(format(value, ".4f"))

def show_reward_mean(
    ckpt_list: list, 
    dataset: str = "openrlhf_collection", 
    length: int = 128, 
    train_evaluation: bool = True
) -> list:
    ckpt_mean_list: list = []
    for ckpt_dir in ckpt_list:
        if train_evaluation:
            rewards_dict: dict = np.load(f"{ckpt_dir}/{dataset}_reward_train.npy", allow_pickle=True).item()
        else:
            rewards_dict: dict = np.load(f"{ckpt_dir}/{dataset}_reward_test.npy", allow_pickle=True).item()
        
        reward_list: list = []
        if train_evaluation:
            for index in range(length):
                reward: torch.Tensor = rewards_dict[f"{index+1}"]
                reward_list.append(reward.mean().item())
        else:
            for index in range(length):
                reward: torch.Tensor = rewards_dict[f"{index+513}"]
                reward_list.append(reward.mean().item())

        rewards_list: np.ndarray = np.array(reward_list)
        ckpt_mean_list.append(show(rewards_list.mean()))
    return ckpt_mean_list

def show_reward_win_rate(
    ckpt_list: list, 
    dataset: str = "openrlhf_collection", 
    length: int = 128, 
    allow_equal: bool = True,
    train_evaluation: bool = True
) -> np.ndarray:
    ckpt_win_rate_list: np.ndarray = np.zeros((len(ckpt_list), len(ckpt_list)))
    
    for index_1, ckpt_dir_index_1 in enumerate(ckpt_list):
        for index_2, ckpt_dir_index_2 in enumerate(ckpt_list):
            
            if index_1 == index_2:
                ckpt_win_rate_list[index_1][index_2] = 0.5
            
            else:
                if train_evaluation:
                    ckpt_index_1_rewards_dict: dict = np.load(f"{ckpt_dir_index_1}/{dataset}_reward_train.npy", allow_pickle=True).item()
                    ckpt_index_2_rewards_dict: dict = np.load(f"{ckpt_dir_index_2}/{dataset}_reward_train.npy", allow_pickle=True).item()
                else:
                    ckpt_index_1_rewards_dict: dict = np.load(f"{ckpt_dir_index_1}/{dataset}_reward_test.npy", allow_pickle=True).item()
                    ckpt_index_2_rewards_dict: dict = np.load(f"{ckpt_dir_index_2}/{dataset}_reward_test.npy", allow_pickle=True).item()
                
                win_times: int = 0
                if train_evaluation:
                    for index in range(length):
                        reward_index_1: torch.Tensor = ckpt_index_1_rewards_dict[f"{index+1}"]
                        reward_index_2: torch.Tensor = ckpt_index_2_rewards_dict[f"{index+1}"]
                        reward_index_2 = reward_index_2.to(reward_index_1.device)

                        if allow_equal:
                            win_times += int((reward_index_1 >= reward_index_2).sum().item())
                        else:
                            win_times += int((reward_index_1 > reward_index_2).sum().item())
                else:
                    for index in range(length):
                        reward_index_1: torch.Tensor = ckpt_index_1_rewards_dict[f"{index+513}"]
                        reward_index_2: torch.Tensor = ckpt_index_2_rewards_dict[f"{index+513}"]
                        reward_index_2 = reward_index_2.to(reward_index_1.device)

                        if allow_equal:
                            win_times += int((reward_index_1 >= reward_index_2).sum().item())
                        else:
                            win_times += int((reward_index_1 > reward_index_2).sum().item())
                
                win_rate = win_times / (length * reward_index_1.shape[0])
                ckpt_win_rate_list[index_1][index_2] = show(win_rate)
    return ckpt_win_rate_list

if __name__ == "__main__":
    ckpt_list_name: list = ["Original", "PPO_1", "OnlineIPO_1", "OnlineIPO_2", "OnlineIPO_3", "OnlineIPO_4", "NashRS_1", "NashRS_2", "NashRS_3", "NashRS_4", "NashRS_5", "NashRS_6", "NashRS_7", "NashRS_8", "NashRS_9", "NashRS_10", "NashRS_11", "NashRS_12"]
    ckpt_list = ["/root/autodl-tmp/ckpt/Eval_openrlhf_Llama-3.2-1B-Instruct_" + name for name in ckpt_list_name]
    logger = get_logger("evaluation_openrlhf_collection", "/root")
    logger.info("-" * 40)
    logger.info("reward mean (train):")
    table_reward_mean = PrettyTable(["reward mean"] + ckpt_list_name)
    table_reward_mean.add_row([""] + show_reward_mean(ckpt_list, train_evaluation=True))
    logger.info(table_reward_mean)
    logger.info("reward mean (test):")
    table_reward_mean = PrettyTable(["reward mean"] + ckpt_list_name)
    table_reward_mean.add_row([""] + show_reward_mean(ckpt_list, train_evaluation=False))
    logger.info(table_reward_mean)
    logger.info("-" * 40)
    logger.info("win rate (train and allow equal):")
    table_win_rate = PrettyTable(["win rate"] + ckpt_list_name)
    win_rate_list = show_reward_win_rate(ckpt_list, allow_equal=True, train_evaluation=True)
    for index_1, name in enumerate(ckpt_list_name):
        win_rate_list_index_1 = [win_rate_list[index_1][index_2] for index_2 in range(len(ckpt_list))]
        table_win_rate.add_row([f"{name}"] + win_rate_list_index_1)
    logger.info(table_win_rate)
    logger.info("-" * 40)
    logger.info("win rate (train and not allow equal):")
    table_win_rate = PrettyTable(["win rate"] + ckpt_list_name)
    win_rate_list = show_reward_win_rate(ckpt_list, allow_equal=False, train_evaluation=True)
    for index_1, name in enumerate(ckpt_list_name):
        win_rate_list_index_1 = [win_rate_list[index_1][index_2] for index_2 in range(len(ckpt_list))]
        table_win_rate.add_row([f"{name}"] + win_rate_list_index_1)
    logger.info(table_win_rate)
    logger.info("-" * 40)
    logger.info("win rate (test and allow equal):")
    table_win_rate = PrettyTable(["win rate"] + ckpt_list_name)
    win_rate_list = show_reward_win_rate(ckpt_list, allow_equal=True, train_evaluation=False)
    for index_1, name in enumerate(ckpt_list_name):
        win_rate_list_index_1 = [win_rate_list[index_1][index_2] for index_2 in range(len(ckpt_list))]
        table_win_rate.add_row([f"{name}"] + win_rate_list_index_1)
    logger.info(table_win_rate)
    logger.info("-" * 40)
    logger.info("win rate (test and not allow equal):")
    table_win_rate = PrettyTable(["win rate"] + ckpt_list_name)
    win_rate_list = show_reward_win_rate(ckpt_list, allow_equal=False, train_evaluation=False)
    for index_1, name in enumerate(ckpt_list_name):
        win_rate_list_index_1 = [win_rate_list[index_1][index_2] for index_2 in range(len(ckpt_list))]
        table_win_rate.add_row([f"{name}"] + win_rate_list_index_1)
    logger.info(f"\n + {table_win_rate}")