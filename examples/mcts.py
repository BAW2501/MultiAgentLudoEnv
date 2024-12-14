import numpy as np
import time
from Ludo.envs.MultiAgentLudoEnv import LudoEnv


EXPLORATION_DEFAULT = 1.41
DEFAULT_TIME_LIMIT = 1.0 # how long to run MCTS agent to make a move
INFINITY = float("inf")


class Node:
    """Represents a node in the Monte Carlo Search Tree"""

    def __init__(self, state, agent_id, parent=None, action=None):
        self.state = state
        self.agent_id = agent_id
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == LudoEnv.NUM_TOKENS

    def get_best_child(self, exploration_constant=EXPLORATION_DEFAULT):
        """Calculates UCB value for node selection"""
        best_child = None
        best_ucb = -INFINITY
        for child in self.children:
            if child.visits == 0:
                ucb = INFINITY
            else:
                ucb = child.value / child.visits + exploration_constant * np.sqrt(
                    np.log(self.visits) / child.visits
                )
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child

    def expand(self):
        for action in range(LudoEnv.NUM_TOKENS):
            new_state = self.state.copy()
            new_env = LudoEnv()
            new_env.board_state = new_state["observation"]["board_state"].copy()
            new_env.dice_roll = new_state["observation"]["last_roll"]
            new_env.round_count = self.state["round_count"]
            new_env.agent_selection = self.agent_id

            new_env._update_game_state(action)  # Corrected line

            new_state = {
                "observation": {
                    "board_state": new_env.board_state,
                    "last_roll": new_env.dice_roll,
                },
                "round_count": new_env.round_count,
            }

            child_node = Node(new_state, new_env.agent_selection, self, action)
            self.children.append(child_node)

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_terminal(self):
        return np.all(
            self.state["observation"]["board_state"][self.agent_id]
            == LudoEnv.FINAL_SQUARE
        )


class MCTS:
    """Implements Monte Carlo Tree Search algorithm"""

    def __init__(
        self, exploration_constant=EXPLORATION_DEFAULT, time_limit=DEFAULT_TIME_LIMIT
    ):
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit

    def search(self, root_state, root_agent_id):
        root_node = Node(root_state, root_agent_id)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = self.select(root_node)
            reward = self.simulate(node)
            node.backpropagate(reward)

        return root_node.get_best_child(0).action  # Exploitation only

    def select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                node.expand()
                return node.children[0]
            else:
                node = node.get_best_child(self.exploration_constant)
        return node

    def simulate(self, node):
        env = self._initialize_simulation_env(node)

        while True:
            if self._is_game_over(env, node):
                return self._calculate_reward(env, node)

            if env.agent_selection == node.agent_id:
                self._process_agent_turn(env)
            else:
                self._process_opponent_turn(env)

    def _initialize_simulation_env(self, node):
        """Initialize a new environment for simulation"""
        env = LudoEnv()
        env.board_state = node.state["observation"]["board_state"].copy()
        env.dice_roll = node.state["observation"]["last_roll"]
        env.round_count = node.state["round_count"]
        env.agent_selection = node.agent_id
        return env

    def _is_game_over(self, env, node):
        """Check if the game has ended"""
        return (
            env.terminations[env.agent_selection]
            or env.truncations[env.agent_selection]
        )

    def _calculate_reward(self, env, node):
        """Calculate the reward for the current game state"""
        if env.agent_selection == node.agent_id:
            return (
                1
                if np.all(env.board_state[env.agent_selection] == LudoEnv.FINAL_SQUARE)
                else 0
            )
        return 0

    def _process_agent_turn(self, env):
        """Process a single turn for the MCTS agent"""
        env.dice_roll = env._roll_dice()
        possible_moves = self._get_possible_moves(env)
        action = self._select_action(possible_moves)
        self._execute_move(env, action)

    def _process_opponent_turn(self, env):
        """Process a single turn for the opponent"""
        env.dice_roll = env._roll_dice()
        possible_moves = self._get_possible_moves(env)
        action = self._select_action(possible_moves)
        self._execute_move(env, action)

    def _get_possible_moves(self, env):
        """Calculate all possible moves for current state"""
        possible_moves = []
        for action in range(LudoEnv.NUM_TOKENS):
            new_pos = env._calculate_new_position(
                env.board_state[env.agent_selection][action], env.dice_roll
            )
            if new_pos != env.board_state[env.agent_selection][action]:
                possible_moves.append(action)
        return possible_moves

    def _select_action(self, possible_moves):
        """Select an action from possible moves"""
        if not possible_moves:
            return 0  # no choice, pass turn
        return np.random.choice(possible_moves)

    def _execute_move(self, env, action):
        """Execute the selected move and update game state"""
        new_pos = env._calculate_new_position(
            env.board_state[env.agent_selection][action], env.dice_roll
        )
        captured = env._check_capture(env.agent_selection, new_pos)
        env.board_state[env.agent_selection][action] = new_pos

        env.roll_again = (
            env.dice_roll == env.DICE_MAX and not env.terminations[env.agent_selection]
        )
        if not env.roll_again:
            env.agent_selection = (env.agent_selection + 1) % env.NUM_PLAYERS


class RandomAgent:
    """Implements a baseline random movement strategy"""

    def __init__(self, agent_id):
        self.agent_id = agent_id

    def select_move(self, env, observation):
        """Select a random valid move from all possible moves"""
        possible_moves = []
        for action in range(LudoEnv.NUM_TOKENS):
            new_pos = env._calculate_new_position(
                env.board_state[env.agent_selection][action], env.dice_roll
            )
            if new_pos != env.board_state[env.agent_selection][action]:
                possible_moves.append(action)

        if not possible_moves:
            return 0  # no choice, pass turn
        return np.random.choice(possible_moves)


def main():
    """Main game loop"""
    env = LudoEnv()
    env.reset()

    # Initialize agents
    mcts_agent = MCTS(time_limit=DEFAULT_TIME_LIMIT)
    random_agents = [RandomAgent(i) for i in range(1, LudoEnv.NUM_PLAYERS)]

    while True:
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print("Game Over!")
            for agent_id, done in env.terminations.items():
                if done:
                    print(f"Agent {agent_id} has won!")
                    env.reset()
                    break
            continue

        if env.agent_selection == 0:
            # MCTS agent's turn
            root_state = {
                "observation": env.observe(0)["observation"],
                "round_count": env.round_count,
            }
            action = mcts_agent.search(root_state, 0)
        else:
            # Random agent's turn
            action = random_agents[env.agent_selection - 1].select_move(
                env, observation
            )

        env.step(action)
        env.render()


if __name__ == "__main__":
    main()
