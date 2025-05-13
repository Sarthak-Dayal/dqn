import numpy as np
import cv2
from collections import deque
import random
from typing import Any, NamedTuple, Union
import torch

def preprocess(obs: np.ndarray):
    # start from 210 x 160 obs, downsample to 110 x 160
    
    # gray scale image
    grey = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    
    # resize to 110 x 84
    resized = cv2.resize(grey, (84, 110), interpolation=cv2.INTER_AREA)
    
    # crop out the center 84 x 84
    start = (110 - 84) // 2
    cropped = resized[start : start + 84, :]
    
    return cropped


class FrameStack:
    def __init__(self, cap: int, device: Union[torch.device, str]):
        self.cap = cap
        self.buffer = deque(maxlen=cap)
        self.device = device
    
    def reset(self, init_frame: np.ndarray):
        first = preprocess(init_frame)
        
        for _ in range(self.cap):
            self.buffer.append(first)
        
        return torch.from_numpy(np.stack(self.buffer, axis=0)).float().to(self.device)

    def add(self, frame: np.ndarray):
        to_add = preprocess(frame)
        
        self.buffer.append(to_add)
        
        return torch.from_numpy(np.stack(self.buffer, axis=0)).float().to(self.device)
    
class Transition(NamedTuple):
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: Any

class ReplayBuffer:
    def __init__(self, cap: int, batch_size: int, device: Union[torch.device, str]):
        self.cap = cap
        self.buffer = deque(maxlen=cap)
        self.batch_size = batch_size
        self.device = device
    
    def add(self, t: Transition):
        self.buffer.append(t)
    
    def sample(self):
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"Requested {self.batch_size} samples, but buffer contains only {len(self.buffer)}")
        
        transitions = random.sample(self.buffer, self.batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)