from setup_gym_env import SnakeEnv


env = SnakeEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    #while True:#not done:
    while not done:
        env.render()
        random_action = env.action_space.sample()
        print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward',reward)