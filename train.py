from utils import ReplayBuffer, FrameStack, Transition
from actor import DQNAgent
import gymnasium as gym
import ale_py
import torch

gym.register_envs(ale_py)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STACKER_CAP = 4
REPLAY_CAP = 10000
BATCH_SIZE = 32
TRAIN_STEPS = 100000
GAMMA = 0.95
eps_min, eps_start, decay_rate = 0.05, 1, 1/TRAIN_STEPS

stacker = FrameStack(STACKER_CAP, DEVICE)
buffer = ReplayBuffer(REPLAY_CAP, BATCH_SIZE, DEVICE)
env = gym.make('ALE/Breakout-v5')
actor = DQNAgent(env.observation_space.shape, env.action_space.n, GAMMA, DEVICE)

obs, info = env.reset()
state = stacker.reset(obs).unsqueeze(0)
total = 0

for i in range(TRAIN_STEPS):
    terminated, truncated = False, False
    eps = max(eps_min, eps_start - i * decay_rate)

    while not terminated and not truncated:
        act = actor.act(state, eps)
        obs, reward, terminated, truncated, info = env.step(act)
        next_state = stacker.add(obs)
        buffer.add(Transition(state.squeeze(0), act, reward, next_state, terminated or truncated))
        total += reward

        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample()
            actor.train_step(batch)

        state = next_state.unsqueeze(0)

    
    obs, info = env.reset()
    state = stacker.reset(obs).unsqueeze(0)
    if i % 100 == 0:
        print(f"Episode: {i}")
        print(f"Total Reward: {total}")
    total = 0

env.close()