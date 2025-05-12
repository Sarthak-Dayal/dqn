from network import QNetwork
import torch
from torch import nn

class DQNAgent:
    def __init__(self, obs_shape, num_actions, gamma):
        self.net = QNetwork(obs_shape, num_actions)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1e-4)
    
    def act(self, obs, epsilon=0.0):
        if torch.rand(1) < epsilon:
            return torch.randint(self.net.out_shape, (1,))
        
        return torch.argmax(self.net(obs), dim=1)

    def train_step(self, batch):
        Q_pred = self.net(torch.stack(batch.state))
        Q_pred = torch.gather(Q_pred, 1, torch.stack(batch.action))
        
        with torch.no_grad():
            nextQ = self.net(torch.stack(batch.next_state))
        
            # if a state is done, do not bootstrap
            # dim 0 is batch dim, dim 1 should be a bunch of Q values for all actions, we want the max
            expected = torch.tensor(batch.reward) + self.gamma * torch.max(nextQ, dim=1).values * (1.0 - torch.tensor(batch.done).float())
        
        loss = self.criterion(Q_pred, expected.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        