import sys
import os

# DRONE_PATH = "<YOUR_ACMPC_FOLDER>/diff_mpc_drones/"


# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import drone
import il_env

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.envs.HoverAviary import HoverAviary
# from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType

from env.ThrustRateAviary import ActionType
from env.HoverAviary import HoverAviary

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('tr') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

state_dim = 20  # 你的状态维度，比如 p,q欧拉,v
state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        self.features_in_dim = feature_dim;

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        self.T = int(os.environ.get("ACMPC_T", 20)) 
        self.n_o = 28
        self.n_output = self.n_o * self.T
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.predictions = th.zeros((self.T, 1, 17)).to(device=self.device)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, self.n_output), nn.Sigmoid()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU()
        )


        self.mpc_env = il_env.IL_Env("drone", mpc_T=self.T)
        self.dx = drone.DroneDx(device=self.device)
        self.u_prev = None

        print(self.policy_net)
        print(self.value_net)

    def forward(self, features: th.Tensor, states: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """ 
        return self.forward_actor(features, states), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, states: th.Tensor) -> th.Tensor:

        states = states.to(self.device).float()
        features_in = features[:, :self.features_in_dim]
        if (states.ndimension() == 1):
            states = th.unsqueeze(states, dim=0)

        # [p, q, v]:
        # states = states[:, 0:10]
        p = states[:, 0:3]
        q = states[:, 3:7]
        v = states[:, 10:13]

        states = th.cat([p, q, v], dim=1)

        # Forward MLP to get cost function for MPC
        sigmoid_cost_all = self.policy_net(features_in)

        # Solve optimization in smaller batches
        n_batch = features.shape[0]

        chunk_length = 1024
        # n_chunks = n_batch // chunk_length + 1

        chunks = th.split(sigmoid_cost_all, chunk_length, dim=0)
        epsilon = 0.1
        range_Q = 100000.0
        range_p = 100000.0
        # range_p_t = 2 * range_Q / 2 * self.dx.mass * 9.806
        range_p_t = 2 * range_Q / 2 * self.dx.mass * 9.8
        n_tau = 14


        if (self.u_prev is None):
            self.u_prev = th.zeros(4, n_batch).to(device=self.device)
            # self.u_prev[0, :] = self.dx.mass * 9.806
            self.u_prev[0, :] = self.dx.mass * 9.8


        # Containers for full solution
        nom_x = th.zeros((n_batch, self.T, self.dx.n_state)).to(device=self.device)
        nom_u = th.zeros((n_batch, self.T, self.dx.n_ctrl)).to(device=self.device)
        idx_start = 0

        for idx, sigmoid_cost in enumerate(chunks):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk
            x_Q = sigmoid_cost[:, :14*self.T].to(device=self.device)  # these are between 0 and 1 right now
            x_p = sigmoid_cost[:, 14*self.T:].to(device=self.device)  # these are between 0 and 1 right now

            q_p = x_Q[:, :3*self.T] * range_Q + epsilon
            q_q = x_Q[:, 3*self.T:7*self.T] * range_Q + epsilon
            q_v = x_Q[:, 7*self.T:10*self.T] * range_Q + epsilon
            q_w = x_Q[:, 10*self.T:13*self.T] * range_Q + epsilon
            q_t = x_Q[:, 13*self.T:14*self.T] * range_Q + epsilon

            p_p = (x_p[:, :3*self.T] - 0.5) * range_p
            p_q = (x_p[:, 3*self.T:7*self.T] - 0.5) * range_p
            p_v = (x_p[:, 7*self.T:10*self.T] - 0.5) * range_p
            p_w = (x_p[:, 10*self.T:13*self.T] - 0.5) * range_p
            p_t = x_p[:, 13*self.T:14*self.T] * range_p_t + epsilon

            u_prev_chunk = self.u_prev[:, idx_start:idx_end]

            _Q = th.zeros(self.T, n_chunk, n_tau, n_tau, device=self.device)
            _p = th.zeros(self.T, n_chunk, n_tau, device=self.device)


            states_chunk = states[idx_start:idx_end, :]

            for i in range(self.T):

                Q_diag_embed_i = th.diag_embed(th.cat([q_p[:, i*3:i*3+3],
                                                         q_q[:, i*4:i*4+4],
                                                         q_v[:,i*3:i*3+3],
                                                         q_t[:,i].unsqueeze(1),
                                                         q_w[:,i*3:i*3+3]], dim=1))


                p_i = th.cat([p_p[:, i*3:i*3+3],
                             p_q[:, i*4:i*4+4],
                             p_v[:, i*3:i*3+3],
                             -p_t[:,i].unsqueeze(1),
                             p_w[:,i*3:i*3+3],
                             ], dim=1)


                _Q[i, :,:,:] = Q_diag_embed_i
                _p[i, :, :] = p_i


            # Run MPC
            nom_x_chunk, nom_u_chunk = self.mpc_env.mpc(
                self.dx, states_chunk, _Q, _p,
                # u_init=train_warmstart[idxs].transpose(0,1),
                u_init=u_prev_chunk,
                # eps_override=0.1,
                lqr_iter_override=1,
            )

            nom_x[idx_start:idx_end, :, :] = nom_x_chunk.transpose(0,1)
            nom_u[idx_start:idx_end, :, :] = nom_u_chunk.transpose(0,1)
            idx_start = idx_end


        self.u_prev = nom_u[:,0,:].transpose(0,1)

        self.predictions = th.cat((nom_x, nom_u), dim=2).detach()


        # Return actions from MPC. These actions will be taken into account to create a gaussian distribution.
        # Units of first control input are thrust normalized by mass
        thrust = nom_u[:, 0, 0]/self.dx.mass
        # The other 3 control inputs are the body rates, in rad/s
        omegas = nom_u[:,0,1:4]

        # Now we normalize the units, since the simulation environment later will unnormalize them by default
        # normalization_max = 8.5 # Max thrust per rotor in Newtons
        # normalization_max = self.dx.mass * 9.8 * 2.25 / 4.0
        # force_mean = (normalization_max * 4 / self.dx.mass) / 2.0
        # force_std = (normalization_max * 4 / self.dx.mass) / 2.0
        thrust_max = self.dx.mass * 9.8 * 2.25 
        thrust_normalized = th.div(thrust, thrust_max).to(device=self.device)

        # print("normalized_thrust_origin")
        # print(thrust_normalized)

        # omega_max = th.Tensor([10.0, 10.0, 4.0]).to(device=self.device)

        # omegas_normalized = th.div(omegas, omega_max).to(device=self.device)

        inputs_normalized = th.cat((thrust_normalized.unsqueeze(1), omegas), dim=1).to(self.device)


        return inputs_normalized


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        return self.value_net(features_in)


class MlpMpcPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            distr_identity = True, # We have a distribution identity such that there is no extra neural network after the MPC output
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

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
    target_reward = 400
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

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##

    test_env = HoverAviary(gui=gui,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            record=record_video)
    test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    state = np.array([info["state"]]) 
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, state=state,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    obs2[3:15],
                                    act2
                                    ]),
                control=np.zeros(12)
                )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()
        
        
if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))