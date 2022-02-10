from stable_baselines3 import PPO
import os
from setup_gym_env import SnakeEnv
import time



models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SnakeEnv()
env.reset()

#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

ckpt = "./models/1644478002/400000"
model= PPO.load(ckpt, verbose=1, tensorboard_log=logdir)

model.set_env(env)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    #env.render()