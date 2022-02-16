# wrong version https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/her/README.md
# right version https://github.com/DLR-RM/stable-baselines3/blob/c41368f2ead24c0cea218164c19e58d48a47422c/docs/modules/her.rst
# her sb3 config https://stable-baselines3.readthedocs.io/en/master/modules/her.html?highlight=HerReplayBuffer#example
# gym robotic https://gym.openai.com/envs/#robotics


import time
import os
import numpy as np

import gym
from stable_baselines3 import HerReplayBuffer, DDPG, SAC
from sb3_contrib.tqc.tqc import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper


models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

#env = gym.make("FetchReach-v1")
env = gym.make("Humanoid-v2")
#env = TimeFeatureWrapper(env)
env.reset()

"""
# config refferences https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/her.yml
FetchReach-v1:
  n_timesteps: !!float 20000
  policy: 'MlpPolicy'
  model_class: 'sac'
  n_sampled_goal: 4
  goal_selection_strategy: 'future'
  buffer_size: 1000000
  ent_coef: 'auto'
  batch_size: 256
  gamma: 0.95
  learning_rate: 0.001
  learning_starts: 1000
  online_sampling: True
  normalize: True

FetchPickAndPlace-v1:
  env_wrapper:
    - sb3_contrib.common.wrappers.TimeFeatureWrapper
    # - utils.wrappers.DoneOnSuccessWrapper:
    #     reward_offset: 0
    #     n_successes: 4
    # - stable_baselines3.common.monitor.Monitor
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  model_class: 'tqc'
  n_sampled_goal: 4
  goal_selection_strategy: 'future'
  buffer_size: 1000000
  batch_size: 1024
  gamma: 0.95
  learning_rate: !!float 1e-3
  tau: 0.05
  policy_kwargs: "dict(n_critics=2, net_arch=[512, 512, 512])"
  online_sampling: True
"""
"""
model = SAC('MultiInputPolicy', env,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy='future',
                #buffer_size=int(1e6),
                #learning_rate=1e-3,
                #gamma=0.95, 
                #batch_size=256,
                online_sampling=True,
                #learning_starts=1000,
                #normalize=True
            ),
            policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
            batch_size=1024,
            gamma=0.95,
            tau=0.05,
            verbose=1,
            tensorboard_log=logdir
            )
"""
model = SAC('MlpPolicy', env,
            policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
            batch_size=1024,
            gamma=0.95,
            tau=0.05,
            verbose=1,
            tensorboard_log=logdir
            )


#ckpt = "./models/1644478002/400000"
#model= PPO.load(ckpt, verbose=1, tensorboard_log=logdir)

#model.set_env(env)

TIMESTEPS = 1e6
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"human-sac")
model.save(f"{models_dir}/{TIMESTEPS}")