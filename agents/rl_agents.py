
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
