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

def compute_P(transition_probs, num_actions, num_states):
    ret = np.zeros((num_states, num_states))
    for s in transition_probs.keys():
        for a in transition_probs[s]:
            for tup in transition_probs[s][a]:
                sp = tup[1]
                p_sasp = tup[0]
                ret[s,sp] += 1.0/num_actions * p_sasp

    return ret



done = False
seed = 1358
env_name = 'WalkFiveStates-v0'
env = init_env(env_name, seed)

num_episodes = 1
G = []
transition_probs = env.env.P
P = compute_P(transition_probs, env.action_space.n, env.observation_space.n)
gamma = .5


# def run_env_episodes(num_episodes):
#     D = np.zeros(env.observation_space.n)
#     total_steps = 0
#     for ep in range(num_episodes):
#         env.reset()
#         done = False
#         while not done:
#             #env.render()
#             state, reward, done, info = env.step(random.randint(0, env.action_space.n - 1))
#             D[state] +=1
#             total_steps +=1
#     #print('discounted rewards:{0}'.format(G))
#     #print ("average score: ", sum(collected_rewards) / num_episodes)
#     print(D / total_steps, total_steps)
#     return np.diag(D / total_steps)

# print('Generate D matrix...')
# D = run_env_episodes(100000)
# print('Done finding D!')

D = np.diag([0.12443139 ,0.24981192 ,0.25088312, 0.25018808 ,0.12468549])

collected_rewards = []
for ep in range(num_episodes):
    ep_rewards = []
    env.reset()
    done = False
    episode_reward = 0
    while not done:
        #env.render()
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


