
from typing import Tuple, Union, List
import numpy as np

from lux.game import Game
from lux.game_map import Cell, GameMap, Position, RESOURCE_TYPES
from lux.game_objects import Unit, City, CityTile
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from agents.base_agent import BaseAgent

class LazyAgent(BaseAgent):
    def get_actions(self, game_state: Game) -> list:
        return []


class TotallyRandomAgent(BaseAgent):

    def get_actions(self, game_state: Game) -> list:
        """
        This method is what changes from agent to agent

        Actions:
             "m u_18 n"                 move u_18 north
             "?"                        pillage road
             "t u_14 u_24 coal 2000"    transfer
             "bcity u_17"               build citytile by u_17

             "bw 1 7"                   build worker in the citytile at (1, 7)
             "bc 3 9"                   build cart in the citytile at (3, 9)
             "r 4 13"                   research in the citytile at (4, 13)
        """
        v_research, v_units, v_citytiles, v_resources = self.get_observation(game_state)  # needed to get active units

        action_vector = []
        for unit in self._active_units:
            action_vector.append(np.random.choice(list(self.ACTION_MAP['unit'].keys())))

        for _, citytile in self._active_citytiles:
            action_vector.append(np.random.choice(list(self.ACTION_MAP['citytile'].keys())))

        # Once all actions have been specified as a vector, convert to interface with game engine
        actions = self._convert_to_actions(action_vector)

        return actions

class RandomContextualAgent(BaseAgent):

    def get_actions(self, game_state: Game) -> list:
        """
        This method is what changes from agent to agent

        Actions:
             "m u_18 n"                 move u_18 north
             "?"                        pillage road
             "t u_14 u_24 coal 2000"    transfer
             "bcity u_17"               build citytile by u_17

             "bw 1 7"                   build worker in the citytile at (1, 7)
             "bc 3 9"                   build cart in the citytile at (3, 9)
             "r 4 13"                   research in the citytile at (4, 13)
        """
        player = game_state.players[self.team]
        v_research, v_units, v_citytiles, v_resources = self.get_observation(game_state)
        t = self.get_observation_as_tensor(game_state)
        step_rewards = self.get_rewards(game_state)  # uc, cc, fg, rp

        # Generate encoded action probabilities vector -----------------------------------------
        # u_probs = np.ones((len(self.ACTION_MAP['unit']),))
        # c_probs = np.ones((len(self.ACTION_MAP['citytile']),))
        # it's ok if they don't add up to 1, they will be normalised later
        u_probs = self._get_action_probs('unit')
        c_probs = self._get_action_probs('citytile')

        ### Warning-based rules  # TODO: in principle, this should be learnt by the agent, but we help it a little for now
        # Agent 0 tried to build unit on tile (3, 27) but unit cap reached. Build more CityTiles!
        o_citytiles = [(city, citytile) for city in player.cities.values() for citytile in city.citytiles]
        if len(o_citytiles) <= len(player.units):
            c_probs[2] = 0  # disable citytile build_worker
            # c_probs[3] = 0  # disable citytile build_worker build_cart
        # This is just a heuristic based on the game rules
        if player.research_points >= GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM']:
            c_probs[0] = 0  # disable research, no need to research more

        ### Generate actions based on context
        action_vector = []

        for unit in self._active_units:
            u_probs_ = np.copy(u_probs)  # Different units get different action probabilities based on context

            ### Warning-based rules    TODO: remove these rules and replace with penalties for RL?
            # Agent 0 tried to build CityTile on existing CityTile; turn 14; cmd: bcity u_3
            # Agent 0 tried to build CityTile on non-empty resource tile; turn 19; cmd: bcity u_3
            if not unit.can_build(game_state.map):
                u_probs_[0] = 0  # NOTE: this is still buggy?

            # Agent 0 tried to move unit u_1 onto opponent CityTile
            # Agent 0 tried to move unit u_1 off map; turn 53; cmd: m u_1 s
            # turn 41; Unit u_3 collided when trying to move s to (4, 29)  (ignore this one, can't do anything about it)
            non_valid_dir_idx = self._get_valid_directions(game_state.map, unit.pos,
                                                           return_inverse=True, return_indices=True)
            for nvdx in non_valid_dir_idx:
                u_probs_[nvdx+1] = 0

            u_probs_ /= u_probs_.sum()  # Normalise probabilities
            action_vector.append(np.random.choice(
                list(self.ACTION_MAP[unit.__class__.__name__.lower()].keys()), p=u_probs_))  # TODO: random actions for now

        c_probs /= c_probs.sum()  # Normalise probabilities
        for _, citytile in self._active_citytiles:
            action_vector.append(np.random.choice(
                list(self.ACTION_MAP[citytile.__class__.__name__.lower()].keys()), p=c_probs))

        # Once all actions have been specified as a vector, convert to interface with game engine
        actions = self._convert_to_actions(action_vector, active_only=True)

        return actions

    def _get_action_probs(self, actor_type):
        if actor_type == 'unit':
            return np.array([20., 0., 2., 2., 2., 2.])  # build_city nothing move(4)
        elif actor_type == 'citytile':
            return np.array([1., 0., 10.])  # research nothing build_worker build_cart
        else:
            raise NotImplementedError


class RandomMaturingAgent(RandomContextualAgent):

    def _get_action_probs(self, actor_type):
        if actor_type == 'unit':
            return self._get_unit_action_probs()
        elif actor_type == 'citytile':
            return self._get_citytile_action_probs()
        else:
            raise NotImplementedError

    def _get_unit_action_probs(self):
        if self.game_state.turn < 30:
            probs = [20., 5., 2., 2., 2., 2.]
        else:
            probs = [20., 1., 2., 2., 2., 2.]
        return np.array(probs)


    def _get_citytile_action_probs(self):
        probs = [1., 0.2, 10.]
        return np.array(probs)
