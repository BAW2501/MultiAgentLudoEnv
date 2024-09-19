from gymnasium.envs.registration import register

register(
    id='Ludo-v0',
    entry_point='ludo.envs:MultiAgentLudoEnv',
)