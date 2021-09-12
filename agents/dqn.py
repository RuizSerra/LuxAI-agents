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

        KERNEL_SIZE = 3
        STRIDE = 2

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=KERNEL_SIZE, stride=STRIDE):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNAgent(BaseAgent):

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(32, interpolation=Image.CUBIC),
                        T.ToTensor()])

    def init_params(self, game_state):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.last_obs = self.get_observation_as_tensor(game_state)
        _, screen_channels, screen_height, screen_width = self.last_obs.shape # (1, 14, 32, 32)  # FIXME

        self.n_actions = 6  # n s e w bcity none

        self.policy_net = DQN(screen_height, screen_width, screen_channels, self.n_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, screen_channels, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.eval()  # ADDED THIS, MAYBE IT'S BAD

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        print('Observation space shape:', self.last_obs.shape)


    def get_observation_as_tensor(self, game_state: Game):
        o = super().get_observation_as_tensor(game_state)
        # Transpose it into torch order (CHW)
        o = o.transpose((2, 0, 1))
        o = torch.from_numpy(o).float()
        # o = resize(o)  # TODO: may need this for different sized game maps
        # Add a batch dimension (BCHW)
        o = o.unsqueeze(0)
        return o

    def get_actions(self, game_state: Game) -> list:

        if game_state.turn == 0:
            self.init_params(game_state)

        self.curr_obs = self.get_observation_as_tensor(game_state)
        self.state_obs = self.curr_obs - self.last_obs
        self.last_obs = self.curr_obs

        # Select action ---------------------------------------
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(self.state_obs).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

        # Convert actions for interface ------------------------------
        my_units = [u for u in self.units if u.team == self.team]
        my_citytiles = [u for u in self.units if u.team == self.team]
        action_vector = [action.item()]*len(my_units)  # all units do the same for now
        action_vector += [1]*len(my_citytiles)  # citytiles do nothing for now
        actions = self._convert_to_actions(action_vector)

        if self.training_mode:
            # TODO: could be that the Agent only outputs action_vector, and conversion is done outside the agent?
            return actions, action_vector

        return actions

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # TODO
        # if any([lambda s: s is None for s in batch.state]):
        #     print("BATCH ISSUE")

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to self.policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self.target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
