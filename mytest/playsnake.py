from stable_baselines3 import PPO
import os
from setup_gym_env import SnakeEnv
import time



models_dir = "./models/1644408901/" + "40000"

env = SnakeEnv()
env.reset()

model = PPO.load(models_dir)

episodes = 500

for episode in range(episodes):
    done = False
    obs = env.reset()
    #while True:#not done:
    while not done:
        env.render()
        action = model.predict(obs)
        print("action",action)
        obs, reward, done, info = env.step(action)
        print('reward',reward)
