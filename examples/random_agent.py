from Ludo.envs import LudoEnv
from Ludo.utils.LudoPygameVisualizer import LudoPygameVisualizer
from time import sleep

RUN_VISUALIZER = True
# cannot close from pygame must ctrl+c from terminal
if __name__ == "__main__":
    env = LudoEnv()
    board = None
    if RUN_VISUALIZER:
        board = LudoPygameVisualizer()
    env.reset(seed=42)

    for agent in env.agent_iter():
        if board is not None:
            board.update(env.board_state)
            sleep(0.5)
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent {agent} has reached the final square.")
            break
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
        env.step(action)
        env.render()
    env.close()
