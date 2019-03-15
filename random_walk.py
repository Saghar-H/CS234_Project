import gym
import gym_walk
import pdb
import numpy as np
import random
from pprint import pprint


def get_discounted_rewards(episode_rewards, gamma):
    discounted_rewards = [0] * (len(episode_rewards) + 1)
    for i in range(len(episode_rewards)-1,-1,-1):
        discounted_rewards[i] = discounted_rewards[i+1] * gamma + episode_rewards[i]
    return discounted_rewards[:-1]
    
def init_env(env_name, seed):
    env = gym.make(env_name)
    env.reset()
    random.seed(seed)
    env.seed(seed)
    return env


done = False
seed = 1358
env_name = 'WalkFiveStates-v0'
env = init_env(env_name, seed)

num_episodes = 10
G = []
transition_probs = env.env.P
gamma = .5

collected_rewards = []
for ep in range(num_episodes):
    ep_rewards = []
    env.reset()
    done = False
    episode_reward = 0
    while not done:
        env.render()
        state, reward, done, info = env.step(random.randint(0, env.action_space.n - 1))
        ep_rewards.append(reward)
        episode_reward += reward
    ep_discountedrewards = get_discounted_rewards(ep_rewards, gamma)
    G.append(ep_discountedrewards)
    collected_rewards.append(episode_reward)
    print ("episode total reward ", episode_reward, " after episode: ", ep)

print('discounted rewards:{0}'.format(G))
print ("average score: ", sum(collected_rewards) / num_episodes)
print("#########")


