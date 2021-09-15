"""
Base Agent that other agents inherit from
"""

from typing import Tuple, Union, List

from lux.game import Game
from lux.game_map import Cell, GameMap, Position, RESOURCE_TYPES
from lux.game_objects import Unit, City, CityTile
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from agents.utils import helpers

import numpy as np

class BaseAgent:
#     UNIT_ACTION_RANGE: Tuple[int] = (-1, 4)
#     CITYTILE_ACTION_RANGE: Tuple[int] = (-1, 2)

    # NOTE: the values (e.g. "move south") separated by spaces are method, argument
    ACTION_MAP = {'unit': {-1: 'build_city',  # TODO: this is for readability, it could just be a list with "build_city" last
                            0: 'DO_NOTHING',
                            1: 'move n',
                            2: 'move s',
                            3: 'move e',
                            4: 'move w',  # TODO: add other actions
                          },
                 'citytile': {-1: 'research',
                               0: 'DO_NOTHING',
                               1: 'build_worker',
                               #2: 'build_cart'
                             }
                 }

    def __init__(self, team=0, training_mode=False, **kwargs):
        self.team = team
        self.training_mode = training_mode
        self.__dict__.update(kwargs)

    def get_cell_as_vector(self, cell: Cell, include_pos=False):
        """
        Output dimensions:
            pos.x            (if include_pos)
            pos.y            (if include_pos)
            wood amount
            coal amount
            uranium amount
            road amount
        """
        crs = [0,0,0]
        if cell.resource:
            crs[list(GAME_CONSTANTS['RESOURCE_TYPES'].values()).index(cell.resource.type)] = float(cell.resource.amount)/800
            assert float(cell.resource.amount)/800 <= 1, "Warning: Cell has more resources than 800!!"
        if include_pos:
            raise NotImplementedError()
        return np.array([*crs,
                         float(cell.road)/6])

    def get_citytile_as_vector(self, citytile: CityTile, city: City, include_pos=False):
        """
        Output dimensions:
            team
            pos.x            (if include_pos)
            pos.y            (if include_pos)
            cooldown/10
            city_id as int   (not used for now)
            fuel
            light_upkeep
        """
        # Using a heuristic for light upkeep:
        #     If the city has one tile, 23/23 = 1
        #     For two tiles, 2*(23-5*1)/2*23 = 0.78
        #     For more tiles it gets more complicated but you get the idea
        light_upkeep_h = float(city.light_upkeep)/(len(city.citytiles)*23)
        assert light_upkeep_h <= 1, "Warning: Light upkeep heuristic is buggy"

        # Heuristic to normalise fuel in city
        #     The amount of fuel in a city is related to the light_upkeep
        #     And required to survive the remaining night turns in the episode
        #     Episodes are 360 in length (add 0.1 to avoid zero division error)
        #     Day+night cycle is 30+10
        #     Finally we make sure it's between 0 and 1, brute force
        fuel_h = (float(city.fuel)/city.light_upkeep) / ((10/40)*(360.1-self.game_state.turn))
        fuel_h = min(fuel_h, 1)
        # if citytile.team == 1:
        #     print(self.game_state.turn, light_upkeep_h, fuel_h)

        if include_pos:
            raise NotImplementedError()
        return np.array([  citytile.team,
                           #citytile.pos.x, citytile.pos.y,
                           float(citytile.cooldown)/10,
                           #int(city.cityid.split('_')[-1]),
                           fuel_h,
                           light_upkeep_h])

    def get_unit_as_vector(self, unit: Unit, include_pos=False):
        """
        Output dimensions:
            team
            pos.x            (if include_pos)
            pos.y            (if include_pos)
            unit type
            cooldown/4
            wood/100
            coal/100
            uranium/100
        """
        if include_pos:
            raise NotImplementedError()
        return np.array([unit.team,
                         # unit.pos.x, unit.pos.y,
                         unit.type,  # TODO: will need this once I incorporate carts
                         float(unit.cooldown)/4,
                         float(unit.cargo.wood)/100,
                         float(unit.cargo.coal)/100,
                         float(unit.cargo.uranium)/100])

    def get_observation_as_tensor(self, game_state: Game):
        """Get representation of observation as HxWxC tensor

        Inspired by https://www.kaggle.com/aithammadiabdellatif/keras-lux-ai-reinforcement-learning
        """
        self.game_state = game_state
        width, height = game_state.map_width, game_state.map_height
        # FIXME: using zeros as default value is not ideal for pos (x,y), could use (-1) or NaN?
        M = np.zeros((height, width, 4))  # cell vector depth
        C = np.zeros((height, width, 4))  # ct vector depth
        U = np.zeros((height, width, 6))  # unit vector depth

        for y in range(height):
            for x in range(width):
                cell = game_state.map.get_cell(x, y)
                M[y, x] = self.get_cell_as_vector(cell)
                # if cell.citytile:  # Could do this here but we need the city as well
                #     C[y, x] = self.get_citytile_as_vector(cell.citytile, city???)

        self.units = []
        self.citytiles = []
        self.cities = []
        for player in game_state.players:
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                # FIXME: if two units in the same citytile, one will overwrite the other in the observation
                U[y, x] = self.get_unit_as_vector(unit)
                self.units.append(unit)
            for city in player.cities.values():
                for citytile in city.citytiles:
                    x, y = citytile.pos.x, citytile.pos.y
                    C[y, x] = self.get_citytile_as_vector(citytile, city)
                    self.citytiles.append(citytile)
                self.cities.append(city)

        return np.dstack([M,U,C])

    def get_observation_as_vectors(self, game_state: Game):
        """
        Returns:
            - units
            - citytiles
            - resources
            - research

        NOTES:
        Observations/updates:
            "rp 1 25"                       research_points player amount
            "r wood 0 5 500"                resource type pos.x pos.y amount
            "r coal 6 0 379"
            "r uranium 6 6 323"
            "u 0 1 u_28 10 10 0 40 0 0"     unit type team id pos.x pos.y cooldown cargo.wood cargo.coal cargo.uranium
            "c 0 c_1 210 23"                city team id fuel light_upkeep
            "ct 0 c_1 3 9 6"                citytile team city.id pos cooldown
            "ccd 0 4 6"                     cellcooldown pos.x pos.y cooldown  (i.e. road or citytile, citytiles have ccd=6)
        """
        # CITY: 'cityid', 'citytiles', 'fuel', 'get_light_upkeep', 'light_upkeep', 'team'
        # CITYTILE: 'build_cart', 'build_worker', 'can_act', 'cityid', 'cooldown', 'pos', 'research', 'team'

        # Collate units and citytiles that can act this turn, in the appropriate vector format
        v_units = []
        v_citytiles = []
        v_research = []

        # TODO: ignoring opponent for now
        # for team, player in enumerate(game_state.players[:1]):
        player = game_state.players[self.team]
        self._active_citytiles = [(city, citytile) for city in player.cities.values() for citytile in city.citytiles if citytile.can_act()]
        self._active_units = [unit for unit in player.units if unit.can_act()]

        # Research
        v_research.append(player.research_points)

        # Units
        for unit in self._active_units:
            v_units.append([
                              # team,
                              # unit.type,  # TODO: will need this once I incorporate carts
                              unit.pos.x, unit.pos.y,
                              unit.cargo.wood, unit.cargo.coal, unit.cargo.uranium,
                              0, 0, 0,  # citytiles padding
            ])

        # Citytiles
        for city, citytile in self._active_citytiles:
            v_citytiles.append([
                                  # team,
                                  # 2,  # i.e. "made up" unit type
                                  citytile.pos.x, citytile.pos.y,
                                  0, 0, 0,  # units padding
                                  int(city.cityid.split('_')[-1]), city.fuel, city.light_upkeep
            ])


        # Doing this here so that it is outside the "for player in players" loop,
        # to collate both players (once I incorporate opponent)
        v_research = np.array(v_research, dtype=np.int16)
        v_units = np.array(v_units, dtype=np.int16)
        v_citytiles = np.array(v_citytiles, dtype=np.int16)

        # Resources relative to this player (based on what's researched, what's nearest to units)
        researched_resources = helpers.get_researched_mask(player.research_points)
        v_r = []
        for r in game_state.map.map:
            for cell in r:
                crs = [0, 0, 0]
                if cell.resource and (cell.resource.type in researched_resources):
                    crs[researched_resources.index(cell.resource.type)] = cell.resource.amount
                    v_r.append([
                                  # -1,  # team (impartial)
                                  # 3,  # "made up" type
                                  cell.pos.x, cell.pos.y,
                                  *crs,
                                  0, 0, 0,  # citytile padding
                                  # cell.road
                                 ])
        v_resources = np.array(v_r, dtype=np.int16)

        # Check that units are still alive, otherwise no need to sort resources
        # TODO: (although citytiles could still make more units later on)
        if not v_units.shape == (0,):
            distances, v_resources = helpers.rank_resources(v_resources, v_units, top_k=10)
            # TODO: not using distances further for now (only for ranking resources)

        return v_research, v_units, v_citytiles, v_resources

    def get_observation(self, game_state: Game) -> Tuple:
        return self.get_observation_as_vectors(game_state)

    def get_rewards(self, game_state: Game) -> Tuple:
        # TODO: in the future, we could include data for both this player and opponent
        player = game_state.players[self.team]
        # Unit count
        uc = len(player.units)
        # City count
        cc = len(player.cities)
        # fuel generation
        fg = sum([c.fuel for c in player.cities.values()])
        # research points
        rp = player.research_points
        return uc, cc, fg, rp

    def _get_valid_directions(self,
                             game_map: GameMap,
                             p0: Position,
                             return_inverse=False, return_indices=False) -> List[str]:
        """
        Check if a planned move is valid or not.

        p0: position planning the move from

        Use cases:
            - Agent 0 tried to move unit u_1 onto opponent CityTile
            - Agent 0 tried to move unit u_1 off map; turn 53; cmd: m u_1 s
            - turn 41; Unit u_3 collided when trying to move s to (4, 29)     (ignore this one, can't do anything about it)
        """
        moves = {'n': (p0.x,   p0.y-1),
                 's': (p0.x,   p0.y+1),
                 'e': (p0.x+1, p0.y  ),
                 'w': (p0.x-1, p0.y  )}

        valid_dirs = list(moves.keys())
        for direction, p1 in moves.items():
            if (p1[0] < 0 or  # TODO: optimise for speed
                p1[1] < 0 or
                p1[0] >= game_map.width or
                p1[1] >= game_map.height):
                valid_dirs.remove(direction)  # Off-map
                continue
            c1 = game_map.get_cell(*p1)
            if c1.citytile and c1.citytile.team != self.team:
                valid_dirs.remove(direction)  # Opponent CityTile

        if return_inverse:
            valid_dirs = set(moves.keys()).difference(set(valid_dirs))

        if return_indices:
            valid_dirs = [list(moves.keys()).index(d)+1 for d in valid_dirs]

        return valid_dirs

    def _convert_to_actions(self, action_vector: list, active_only=False) -> list:
        """
        https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
        """

        units = self._active_units if active_only else [u for u in self.units if u.team == self.team]
        citytiles = [c[1] for c in self._active_citytiles] if active_only \
                     else [c for c in self.citytiles if c.team == self.team]

        all_actors = units + citytiles

        actions = []
        for a in action_vector:
            try:
                actor = all_actors.pop(0)
            except IndexError:
                # print('Action/actors mismatch')
                break
            s = self.ACTION_MAP[actor.__class__.__name__.lower()][a].split()
            if s[0] == 'DO_NOTHING':
                continue
            elif len(s) > 1:
                action = getattr(actor, s[0])(*s[1:])
            else:
                action = getattr(actor, s[0])()
            actions.append(action)

        return actions
