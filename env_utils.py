import gym_2048
import gym
from gym.envs.toy_text import taxi

import numpy as np
import random
from boyan_exp import BOYAN_MDP
from compute_utils import get_discounted_return
import gym_walk
import pudb
import pdb

def init_env(env_name, seed):
    if 'Walk' in env_name:
        env = gym.make(env_name)
    elif 'Boyan' in env_name:
        boyan_mdp = BOYAN_MDP('boyan_mdp.png')
        env = boyan_mdp.env
    elif '2048' in env_name:
        env = gym.make('2048-v0')
    elif 'Taxi' in env_name:
        #env = gym.make('Taxi-v2')
        env = taxi.TaxiEnv()
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

'''
Use this when generating 2048 environment traces.
'''
def run_env_episodes_2048(env, config, mode):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    R = np.zeros(env.observation_space.n)
    trajectories = []
    Gs = []
    total_steps = 0
    num_episodes = config.num_train_episodes if mode == 'train' else config.num_test_episodes
    env.render()
    for ep in range(num_episodes):
        trajectories.append([])
        cur_state = env.reset()
        done = False
        ep_rewards = []
        ep_states = []
        done = False
        while not done:
            action = env.np_random.choice(range(4),1).item()
            next_state, reward, done, info = env.step(action)
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


'''
Use this when generating Taxi environment traces.
'''
def run_env_episodes_taxi(env, config, mode):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    R = np.zeros(env.observation_space.n)
    trajectories = []
    Gs = []
    total_steps = 0
    num_episodes = config.num_train_episodes if mode == 'train' else config.num_test_episodes
    env.render()
    ep = 0
    #for ep in range(num_episodes):

    while ep <= num_episodes:
        #trajectories.append([])
        cur_state = env.reset()
        done = False
        ep_rewards = []
        ep_states = []
        traj = []
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            traj.append((cur_state, reward, next_state, done))
            D[cur_state] += 1
            total_steps += 1
            ep_rewards.append(reward)
            ep_states.append(cur_state)
            cur_state = next_state
        if len(ep_states) <= 500 :
            trajectories.append(traj)
            ep += 1
            ep_discountedrewards = get_discounted_return(ep_rewards, config.gamma)
            Gs.append(ep_discountedrewards)

            for i in range(len(ep_states)):
                V[ep_states[i]] += ep_discountedrewards[i]
                R[ep_states[i]] += ep_rewards[i]

    return np.diag(D / total_steps), V / D, trajectories, Gs, R/D
    
def taxi_env_features(P):
    locs = [(0,0), (0,4), (4,0), (4,3)]
    Phi = np.zeros((500, 25))
    states = list(set(P.keys()))
    for i in range(500):
        state = states[i]
        feature = np.zeros((5,5))
        taxirow, taxicol, passloc, destidx = taxi_state_decode(state)
        feature[taxirow, taxicol] += 1
        if passloc == 4:
            feature[taxirow, taxicol] += 1/4
        else:
            feature[locs[passloc][0], locs[passloc][1]] += 1/4
        feature[locs[destidx][0], locs[destidx][1]] += 1/2 
        feature = feature.flatten()
        Phi[i, :] = feature
    return Phi

def taxi_state_decode(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)
    