import gymnasium as gym
import ale_py
from utils import FrameStack
STACKER_CAP = 4

gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5')
stacker = FrameStack(STACKER_CAP)

obs, info = env.reset()
state = stacker.reset(obs)

for i in range(10000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    state = stacker.add(obs)
    env.render(render_mode='human')
    
    if terminated or truncated:
        obs, info = env.reset()
        state = stacker.reset(obs)

env.close()