from stable_baselines3 import PPO
import os
from setup_gym_env import SnakeEnv
import time



#models_dir = "./models/1644408901/" + "40000"
#models_dir = "./models/1644462865/" + "120000"
#models_dir = "./models/1644466638/" + "100000"
models_dir = "./models/1644467633/" + "970000"
env = SnakeEnv()
env.reset()

model = PPO.load(models_dir)

episodes = 10


# snake doesn't known where itself
for episode in range(episodes):
    done = False
    obs = env.reset()
    #while True:#not done:
    while not done:
        action, _states = model.predict(obs)
        #print("action",action)
        obs, reward, done, info = env.step(action)
        #print('reward',reward)
        if done == True:
            print(done)
        env.render()
