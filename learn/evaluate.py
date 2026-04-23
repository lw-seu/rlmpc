import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.enums import ObservationType

from env.HoverAviary import HoverAviary
from env.ThrustRateAviary import ActionType

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_MODEL_PATH = "/home/lw/results/save-04.23.2026_17.18.46/best_model.zip"
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('tr') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

def run_eval(model_path=DEFAULT_MODEL_PATH, 
             record_video=DEFAULT_RECORD_VIDEO,
             output_folder=DEFAULT_OUTPUT_FOLDER,
             colab=DEFAULT_COLAB,
             plot=True):
    
     # 加载模型
    # filename = "/home/lw/results/save-04.22.2026_17.44.06"
    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    # else:
    #     print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(model_path)
    
    # 测试环境
    test_env = HoverAviary(gui=DEFAULT_GUI, 
                           obs=DEFAULT_OBS, 
                           act=DEFAULT_ACT, record=record_video)
    test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder=output_folder,
                colab=colab
                )

    # 评估
    mean_reward, std_reward = evaluate_policy(model, 
                                              test_env_nogui, 
                                              n_eval_episodes=10)
    print("\nMean reward:", mean_reward, "±", std_reward)
    
    

    obs, info = test_env.reset(seed=42, options={})
    state = np.array([info["state"]]) 
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, drone_state=state,
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
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate HoverAviary RL model')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str,
                        help='Path to the saved model .zip')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder to save videos or logs')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=bool,
                        help='Whether to use PyBullet GUI during evaluation')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=bool,
                        help='Whether to record video of evaluation')
    ARGS = parser.parse_args()

    run_eval(model_path=ARGS.model_path,
             output_folder=ARGS.output_folder)