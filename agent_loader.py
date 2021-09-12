"""
Agent loader

Usage:

>>> from agent_loader import AgentLoader
>>> from agents.rl_agents import RLAgent
>>> my_agent = AgentLoader(agent_class=RLAgent).game_loop
>>> env = make("lux_ai_2021")
>>> steps = env.run([my_agent, "simple_agent"])
>>> env.render(mode="ipython", width=1200, height=800)
"""

# LuxAI deps
import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from agents.random_agents import LazyAgent

class AgentLoader:

    def __init__(self, agent_class=LazyAgent):
        self.game_state = None
        self.agent_class = agent_class
        self.agent = None

    def game_loop(self, observation, configuration):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player
            self.agent = self.agent_class(team=observation.player)
        else:
            self.game_state._update(observation["updates"])

        actions = self.agent.get_actions(self.game_state)
        return actions
