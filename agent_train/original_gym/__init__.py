from gymnasium.envs.registration import register

register(
    id="DroneWorld-v0",
    entry_point="original_gym.gym_env:DroneEnv",
)