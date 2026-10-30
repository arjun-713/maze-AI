"""Your AI workspace. No policy, Q-table, network, or learning algorithm is supplied."""

from maze_game.training import AgentDecision, Transition


class Redbull:
    name = "Redbull"
    ready = False  # Set True after you implement act(), then restart the studio.

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        # TODO: Create your own agent state here.

    def act(self, observation: dict) -> AgentDecision:
        """Choose up/right/down/left from the public observation only.

        Return AgentDecision(action, debug={}). Optional debug keys understood
        by the studio are q_values, action_values, and action_probabilities;
        each maps action names to numbers. Other JSON debug data is allowed.
        """
        raise NotImplementedError("Write your own action-selection logic in Redbull/agent.py")

    def on_episode_start(self, observation: dict) -> None:
        """Optional: reset episode memory here; preserve learned state as you choose."""
        pass

    def on_transition(self, transition: Transition) -> None:
        """Optional: implement your own learning update from this action's feedback."""
        pass
