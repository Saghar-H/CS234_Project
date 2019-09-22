from gym.envs.registration import register


# classics
register(
    id='Taxi-v4',
    entry_point='gym_taxi.envs:TaxiEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=50,
)

