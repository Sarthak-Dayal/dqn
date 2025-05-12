import torch
from torch import nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.in_shape = obs_shape
        self.out_shape = num_actions
        
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, self.out_shape)
        )
        
    def forward(self, input):
        Qs = self.net(input)
        return Qs