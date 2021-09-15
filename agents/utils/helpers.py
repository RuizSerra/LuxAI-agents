
from lux.game import Game
from lux.game_map import Cell, GameMap, Position, RESOURCE_TYPES
from lux.game_objects import Unit, City, CityTile
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

import numpy as np

# Helper functions used by agent

def get_researched_mask(research_points):
    """Return a mask of available resources"""
    mask = [Constants.RESOURCE_TYPES.WOOD]
    if research_points >= GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['URANIUM']:
        mask += [Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM]
    elif research_points >= GAME_CONSTANTS['PARAMETERS']['RESEARCH_REQUIREMENTS']['COAL']:
        mask += [Constants.RESOURCE_TYPES.COAL]
    return mask

def closest_node(node, nodes):
    """Credit @glmcdona
    https://www.kaggle.com/glmcdona/lux-ai-deep-reinforcement-learning-ppo-example
    https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    """
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def min_distance(node, nodes):
    assert node.shape == (2,), print(node.shape)
    assert nodes.shape[1] == 2
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.min(dist_2)

def rank_resources(resources: np.ndarray,
                   units: np.ndarray,
                   top_k: int = None,
                   d_thd: int = None) -> tuple:
    """Given a list of units, rank the list of resources by proximity to any unit.

    e.g.

    >>> units = np.array([[  4,  27, 100,   0,   0,   0,   0,   0],])

    >>> resources = np.array([[  4,  27, 238,   0,   0,   0,   0,   0],
                              [  4,  11, 209,   0,   0,   0,   0,   0],
                              [  4,  28, 238,   0,   0,   0,   0,   0],
                              [  5,  11, 126,   0,   0,   0,   0,   0],
                              [  3,  28,  32,   0,   0,   0,   0,   0],])

    >>> d, r = rank_resources(resources, units)

    If top_k is provided, return only the top K resources by distance
    If d_thd is provided, return only the resources within thd distance
    If both are provided, top K takes precedence
    """
    distances = [min_distance(resource, units[:, :2]) for resource in resources[:, :2]] # TODO: this will break when I add team/unit type to input vectors
    to_be_sorted = np.c_[distances, resources]
    done_sorted = to_be_sorted[np.argsort(to_be_sorted[:, 0])]
    if top_k:
        return done_sorted[:top_k, 0], done_sorted[:top_k, 1:]
    elif d_thd:
        return done_sorted[np.where(done_sorted[:, 0] <= d_thd)][:, 0], done_sorted[np.where(done_sorted[:, 0] <= d_thd)][:, 1:]
    return done_sorted[:, 0], done_sorted[:, 1:]  # distances, resources



# -----------------------------------------------------------------------------
# Some helper functions (not used by agent, just for EDA)

def find_by_type(game_state, type_name='resource'):
    """
    type_name (list): 'resource', 'citytile', 'road'
    """
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if getattr(cell, type_name):
                resource_tiles.append(cell)
    return resource_tiles

# the next snippet finds the closest resources that we can mine given position on a map
def find_closest_resources(pos, player, resource_tiles):
    closest_dist = math.inf
    closest_resource_tile = None
    for resource_tile in resource_tiles:
        # we skip over resources that we can't mine due to not having researched them
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile
