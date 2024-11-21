from gymnasium.envs.registration import register

register(
    id="SantaFe_GYM/GridWorld-v0",
    entry_point="SantaFe_GYM.envs:GridWorldEnv",
)
