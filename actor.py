from network import QNetwork
import torch
from torch import nn

class DQNAgent:
    def __init__(self, obs_shape, num_actions, gamma, device):
        self.policy_net = QNetwork(obs_shape, num_actions).to(device)
        self.target_net = QNetwork(obs_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.device = device
        self.gamma = gamma
        self.criterion = nn.SmoothL1Loss()
        self.optim = torch.optim.RMSprop(self.policy_net.parameters(), lr=1e-4)
    
    @torch.no_grad()
    def act(self, obs, epsilon=0.0):
        if torch.rand(1) < epsilon:
            return torch.randint(self.policy_net.out_shape, (1,)).to(self.device)
        
        return torch.argmax(self.policy_net(obs), dim=1).to(self.device)

    def train_step(self, batch):
        Q_pred = self.policy_net(batch.state)
        Q_pred = torch.gather(Q_pred, 1, batch.action)
        
        with torch.no_grad():
            nextQ = self.target_net(batch.next_state).max(1)[0]
        
            # if a state is done, do not bootstrap
            # dim 0 is batch dim, dim 1 should be a bunch of Q values for all actions, we want the max
            expected = batch.reward + self.gamma * nextQ * (1.0 - batch.done)
        
        loss = self.criterion(Q_pred, expected.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optim.step()
        return loss.item()
