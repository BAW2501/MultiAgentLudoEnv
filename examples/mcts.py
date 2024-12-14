import numpy as np
import time
from Ludo.envs.MultiAgentLudoEnv import LudoEnv


class Node:
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

    def select_best_child(self, exploration_constant=1.41):
        best_child = None
        best_ucb = -float("inf")
        for child in self.children:
            if child.visits == 0:
                ucb = float("inf")
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
    def __init__(self, exploration_constant=1.41, time_limit=1.0):
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit

    def search(self, root_state, root_agent_id):
        root_node = Node(root_state, root_agent_id)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = self.select(root_node)
            reward = self.simulate(node)
            node.backpropagate(reward)

        best_child = root_node.select_best_child(0)  # Exploitation only
        return best_child.action

    def select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                node.expand()
                return node.children[0]
            else:
                node = node.select_best_child(self.exploration_constant)
        return node

    def simulate(self, node):

        env = LudoEnv()
        env.board_state = node.state["observation"]["board_state"].copy()
        env.dice_roll = node.state["observation"]["last_roll"]
        env.round_count = node.state["round_count"]
        env.agent_selection = node.agent_id

        while True:

            if (
                env.terminations[env.agent_selection]
                or env.truncations[env.agent_selection]
            ):
                if env.agent_selection == node.agent_id:

                    return (
                        1
                        if np.all(
                            env.board_state[env.agent_selection] == LudoEnv.FINAL_SQUARE
                        )
                        else 0
                    )  # return 1 if win else 0
                else:
                    return 0  # not our turn so no reward
            if env.agent_selection == node.agent_id:

                env.dice_roll = env._roll_dice()

                possible_moves = []
                for action in range(LudoEnv.NUM_TOKENS):

                    new_pos = env._calculate_new_position(
                        env.board_state[env.agent_selection][action], env.dice_roll
                    )
                    if new_pos != env.board_state[env.agent_selection][action]:

                        possible_moves.append(action)

                if not possible_moves:

                    action = 0  # no choice pass turn
                else:
                    action = np.random.choice(possible_moves)

                new_pos = env._calculate_new_position(
                    env.board_state[env.agent_selection][action], env.dice_roll
                )
                captured = env._check_capture(env.agent_selection, new_pos)
                env.board_state[env.agent_selection][action] = new_pos

                env.roll_again = (
                    env.dice_roll == env.DICE_MAX
                    and not env.terminations[env.agent_selection]
                )
                if not env.roll_again:
                    env.agent_selection = (env.agent_selection + 1) % env.NUM_PLAYERS

            else:

                env.dice_roll = env._roll_dice()

                possible_moves = []
                for action in range(LudoEnv.NUM_TOKENS):

                    new_pos = env._calculate_new_position(
                        env.board_state[env.agent_selection][action], env.dice_roll
                    )
                    if new_pos != env.board_state[env.agent_selection][action]:

                        possible_moves.append(action)

                if not possible_moves:

                    action = 0  # no choice pass turn
                else:
                    action = np.random.choice(possible_moves)

                new_pos = env._calculate_new_position(
                    env.board_state[env.agent_selection][action], env.dice_roll
                )
                captured = env._check_capture(env.agent_selection, new_pos)
                env.board_state[env.agent_selection][action] = new_pos

                env.roll_again = (
                    env.dice_roll == env.DICE_MAX
                    and not env.terminations[env.agent_selection]
                )
                if not env.roll_again:
                    env.agent_selection = (env.agent_selection + 1) % env.NUM_PLAYERS


class RandomAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def get_action(self, env, observation):

        possible_moves = []
        for action in range(LudoEnv.NUM_TOKENS):

            new_pos = env._calculate_new_position(
                env.board_state[env.agent_selection][action], env.dice_roll
            )
            if new_pos != env.board_state[env.agent_selection][action]:

                possible_moves.append(action)

        if not possible_moves:

            return 0  # no choice pass turn
        else:
            return np.random.choice(possible_moves)


# Example Usage
env = LudoEnv()
env.reset()

# Initialize agents
mcts_agent = MCTS(time_limit=1.0)
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
            "observation": env.observe(0)["observation"],  # Corrected line
            "round_count": env.round_count,
        }
        action = mcts_agent.search(root_state, 0)
    else:
        # Random agent's turn
        action = random_agents[env.agent_selection - 1].get_action(env, observation)

    env.step(action)
    env.render()
