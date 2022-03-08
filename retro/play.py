import time
from setupEnv import StreetFighter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

_model = "./train_SF/best_model_5000000.zip"
model = PPO.load(_model)

env = StreetFighter()
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')


# Reset game to starting state
obs = env.reset()
# Set flag to flase
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        action, _  = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #time.sleep(0.01)