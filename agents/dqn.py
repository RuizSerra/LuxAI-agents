"""
source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?highlight=reinforcement
"""

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from typing import Tuple, Union, List

from lux.game import Game
from lux.game_map import Cell, GameMap, Position, RESOURCE_TYPES
from lux.game_objects import Unit, City, CityTile
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from agents.base_agent import BaseAgent

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNAgent(BaseAgent):

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(32, interpolation=Image.CUBIC),
                        T.ToTensor()])

    def get_actions(self, game_state: Game) -> list:

        o = self.get_observation_as_tensor(game_state)
        # Transpose it into torch order (CHW)
        o = o.transpose((2, 0, 1))
        o = torch.from_numpy(o)
        # o = resize(o)  # TODO: may need this for different sized game maps
        # Add a batch dimension (BCHW)
        o = o.unsqueeze(0)

        if game_state.turn == 1:
            print('Observation space shape:', o.shape)

        actions = []

        return actions
