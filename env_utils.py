import gym
import numpy as np
import random
from boyan_exp import BOYAN_MDP
from compute_utils import get_discounted_return
import gym_walk

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

def run_env_episodes(env, config):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    R = np.zeros(env.observation_space.n)
    trajectories = []
    Gs = []
    total_steps = 0
    for ep in range(config.num_episodes):
        trajectories.append([])
        cur_state = env.reset()
        done = False
        ep_rewards = []
        ep_states = []
        #Breaking change: for non gym-walk, cur_state-1 and next_state-1 should be changed to cur_state and next_state.
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

    #print('Monte Carlo D:{0}'.format(D * 1.0 / total_steps, total_steps))
    #print('Monte Carlo V:{0}'.format(V * 1.0 / D))
    #print('----------Trajectories---------')
    #print(trajectories[0])
    return np.diag(D / total_steps), V / D, trajectories, Gs, R/D