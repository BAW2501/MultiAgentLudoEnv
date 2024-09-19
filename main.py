from Ludo.envs import LudoEnv

if __name__ == "__main__":
    env = LudoEnv()
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent {agent} has reached the final square.")
            break
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
        env.step(action)
        env.render()
    env.close()