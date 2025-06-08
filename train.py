from utils import ReplayBuffer, FrameStack, Transition
from actor import DQNAgent
import gymnasium as gym
import ale_py
import torch
from gymnasium.wrappers import AtariPreprocessing
gym.register_envs(ale_py)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STACKER_CAP = 4
REPLAY_CAP = 1000000
LEARNING_STARTS = 50000
BATCH_SIZE = 32
TRAIN_STEPS = 50_000_000
ANNEAL_END = 1_000_000
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
eps_min, eps_start, decay_rate = 0.05, 1, 1/ANNEAL_END

stacker = FrameStack(STACKER_CAP, DEVICE)
buffer = ReplayBuffer(REPLAY_CAP, BATCH_SIZE, DEVICE)
env = gym.make('ALE/Breakout-v5')
env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=False, scale_obs=False)

actor = DQNAgent(env.observation_space.shape, env.action_space.n, GAMMA, DEVICE)

obs, info = env.reset()
state = stacker.reset(obs).unsqueeze(0)
total = 0
ep_step = 0

for i in range(TRAIN_STEPS):
    terminated, truncated = False, False
    eps = max(eps_min, eps_start - i * decay_rate)

    act = actor.act(state, eps)

    obs, reward, terminated, truncated, info = env.step(act.item())
    next_state = stacker.add(obs)
    buffer.add(Transition(state.squeeze(0), act, reward, next_state, terminated or truncated))
    total += reward

    if i >= LEARNING_STARTS:
        batch = buffer.sample()
        
        train_batch = Transition(batch.state.float() / 255.0, batch.action, batch.reward, batch.next_state.float() / 255.0, batch.done)
        loss = actor.train_step(train_batch)
        if i % 1000 == 0:
            print(f"Step {i}: Epsilon {eps}")
            print(f"Loss {i}: {loss}")

    state = next_state.unsqueeze(0)
    
    if i % TARGET_UPDATE_FREQ == 0:
        actor.target_net.load_state_dict(actor.policy_net.state_dict())
    
    if ep_step % 100 == 0:
        print(f"Step {ep_step}")
        print(f"Reward so far {total}")
    
    ep_step += 1

    if terminated or truncated:
        ep_step = 0
        obs, info = env.reset()
        state = stacker.reset(obs).unsqueeze(0)
        print(f"Episode: {i}")
        print(f"Total Reward: {total}")
        total = 0


env.close()
