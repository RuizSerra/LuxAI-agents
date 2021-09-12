"""
Reinforcement Learning agents
"""

from typing import Tuple, Union, List

from lux.game import Game
from lux.game_map import Cell, GameMap, Position, RESOURCE_TYPES
from lux.game_objects import Unit, City, CityTile
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

import torch
import torch.nn as nn

from agents.base_agent import BaseAgent

class RLAgent(BaseAgent):

    def get_actions(self, game_state: Game) -> list:
        o = self.get_observation_as_tensor(game_state)

        if game_state.turn == 1:
            print(o.shape)

        # TODO: "research per player" as observation?

        # Once all actions have been specified as a vector, convert to interface with game engine
#         actions = self._convert_to_actions(action_vector)
        return []

        return actions
