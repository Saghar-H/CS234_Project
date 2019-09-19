import gym
import numpy as np
import random
from boyan_exp import BOYAN_MDP
from compute_utils import get_discounted_return
import gym_walk
import pudb

def init_env(env_name, seed):
    if env_name == 'RandomWalk-v0':
        env = gym.make('RandomWalk-v0')
    else:
        boyan_mdp = BOYAN_MDP('boyan_mdp.png')
        env = boyan_mdp.env
    env.reset()
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    return env

'''
Current random walk generates states from 1 ... last state, need to 
substract by 1 when adding to the lists.
'''
def run_env_episodes_walk(env, config, mode):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    R = np.zeros(env.observation_space.n)
    trajectories = []
    Gs = []
    total_steps = 0
    num_episodes = config.num_train_episodes if mode == 'train' else config.num_test_episodes
    for ep in range(num_episodes):
        trajectories.append([])
        cur_state = env.reset()
        done = False
        ep_rewards = []
        ep_states = []
        while not done:
            next_state, reward, done, info = env.step(random.randint(0, env.action_space.n - 1))
            trajectories[ep].append((cur_state-1, reward, next_state-1, done))
            D[cur_state-1] += 1
            total_steps += 1
            ep_rewards.append(reward)
            ep_states.append(cur_state-1)
            cur_state = next_state

        ep_discountedrewards = get_discounted_return(ep_rewards, config.gamma)
        Gs.append(ep_discountedrewards)

        for i in range(len(ep_states)):
            V[ep_states[i]] += ep_discountedrewards[i]
            R[ep_states[i]] += ep_rewards[i]

    return np.diag(D / total_steps), V / D, trajectories, Gs, R/D

'''
Use this when generating Boyan environment traces.
'''
def run_env_episodes_boyan(env, config, mode):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    R = np.zeros(env.observation_space.n)
    trajectories = []
    Gs = []
    total_steps = 0
    num_episodes = config.num_train_episodes if mode == 'train' else config.num_test_episodes

    for ep in range(num_episodes):
        trajectories.append([])
        cur_state = env.reset()
        done = False
        ep_rewards = []
        ep_states = []

        while not done:
            next_state, reward, done, info = env.step(random.randint(0, env.action_space.n - 1))
            trajectories[ep].append((cur_state, reward, next_state, done))
            D[cur_state] += 1
            total_steps += 1
            ep_rewards.append(reward)
            ep_states.append(cur_state)
            cur_state = next_state

        ep_discountedrewards = get_discounted_return(ep_rewards, config.gamma)
        Gs.append(ep_discountedrewards)

        for i in range(len(ep_states)):
            V[ep_states[i]] += ep_discountedrewards[i]
            R[ep_states[i]] += ep_rewards[i]

    return np.diag(D / total_steps), V / D, trajectories, Gs, R/D