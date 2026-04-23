from mlp_mpc_policy import CustomNetwork
from mlp_mpc_policy import MlpMpcPolicy

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.enums import ObservationType
from env.ThrustRateAviary import ActionType
from env.HoverAviary import HoverAviary

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('tr') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_OUTPUT_FOLDER = 'results'

state_dim = 20  # 你的状态维度，比如 p,q欧拉,v
state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

def run(output_folder=DEFAULT_OUTPUT_FOLDER, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')


    train_env = make_vec_env(HoverAviary,
                                env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                n_envs=1,
                                seed=0
                                )

    eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    # model = PPO(MlpMpcPolicy,
    #             train_env,
    #             # tensorboard_log=filename+'/tb/',
    #             verbose=1)
    model = PPO(
        MlpMpcPolicy,
        train_env,
        verbose=1,
        rollout_buffer_kwargs=dict(state_space=state_space)
    )

    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = 450
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(4e4) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0] 
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.show()

if __name__ == "__main__":
    run()