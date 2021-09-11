# for kaggle-environments
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position, GameMap
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import sys

import numpy as np

from agents.rl_agents import RLAgent

myagent = None
# myagent = TotallyRandomAgent(team=0)
# myagent = RandomContextualAgent(team=0)
myagent = RLAgent(team=0)
game_state = None

def agent(observation, configuration):
    global game_state
    global myagent

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    if not myagent:
        # Init default agent if no agent was defined outside the function
        myagent = RandomContextualAgent(team=observation.player)

    actions = myagent.get_actions(game_state)

    return actions
