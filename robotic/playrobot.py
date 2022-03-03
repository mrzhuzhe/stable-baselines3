import os
import numpy as np
import time

import gym
from stable_baselines3 import DDPG, SAC
import cv2
from sb3_contrib.common.wrappers import TimeFeatureWrapper 

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

#if not os.path.exists(models_dir):
#	os.makedirs(models_dir)

#if not os.path.exists(logdir):
#	os.makedirs(logdir)

#env = gym.make("FetchReach-v1")
env = gym.make("FetchPickAndPlace-v1")
#env = gym.make("Humanoid-v2")
env = TimeFeatureWrapper(env)
env.reset()

env.reset()


ckpt = "./models/TQC/10000000"
#ckpt = "./models/human-sac/1000000.0"
model= SAC.load(ckpt, env=env, verbose=1, tensorboard_log=logdir)

episodes = 10
# snake doesn't known where itself
for episode in range(episodes):
    done = False
    obs = env.reset()
    #while True:#not done:
    while not done:
        action, _states = model.predict(obs)
        print("action",action)
        obs, reward, done, info = env.step(action)
        print('reward',reward)
        #if done == True:
            #print(done)
        env.render()
env.close()