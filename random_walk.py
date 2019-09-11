import gym
import gym_walk
import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
from pprint import pprint
from grid_search_utils import find_optimal_lambda_grid_search, find_adaptive_optimal_lambda_grid_search, draw_optimal_lambda_grid_search, draw_box_grid_search
from lstd_algorithms import LSTD_algorithm, Adaptive_LSTD_algorithm, Adaptive_LSTD_algorithm_batch, Adaptive_LSTD_algorithm_batch_type2, compute_CV_loss
from compute_utils import get_discounted_return, compute_P
from Config import Config
#import pudb
################  Parameters #################
done = False
log_events = True

if log_events:
    from tensorboard_utils import Logger
        
config = Config(
    seed = 1356,
    env_name = 'WalkFiveStates-v0',
    num_features = 10,
    num_states = 5,
    num_episodes = 10000,
    A_inv_epsilon = 1e-3,
    gamma = 0.5,
    default_lambda = 0.5,
    lr = .01,
    use_adaptive_lambda = True,
    grad_clip_norm = 10,
    compute_autograd = False,
    use_adam_optimizer = True,
    batch_size = 8,
)

##########################################################

def init_env(env_name, seed):
    env = gym.make(env_name)
    env.reset()
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    return env

def run_env_episodes(num_episodes):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    R = np.zeros(env.observation_space.n)
    trajectories = {}
    Gs = {}
    total_steps = 0
    for ep in range(num_episodes):
        trajectories[ep] = []
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
        Gs[ep] = ep_discountedrewards

        for i in range(len(ep_states)):
            V[ep_states[i]] += ep_discountedrewards[i]
            R[ep_states[i]] += ep_rewards[i]

    #print('Monte Carlo D:{0}'.format(D * 1.0 / total_steps, total_steps))
    #print('Monte Carlo V:{0}'.format(V * 1.0 / D))
    #print('----------Trajectories---------')
    #print(trajectories[0])
    return np.diag(D / total_steps), V / D, trajectories, Gs, R/D

env = init_env(config.env_name, config.seed)

transition_probs = env.env.P
print("###############Transition Probabilities####################")
print(transition_probs)
print('Generate Monte Carlo Estimates of D and V...')
D, V, trajectories, Gs, R = run_env_episodes(config.num_episodes)
print('Done finding D and V!')
Phi = np.random.rand(config.num_states, config.num_features)
# D = np.diag([0.12443139 ,0.24981192 ,0.25088312, 0.25018808 ,0.12468549])
# V = np.array([0, 0.01776151, 0.071083, 0.26708894 ,1])

'''
Now compute the MRP value of P: P(s'|s)
'''


P = compute_P(transition_probs, env.action_space.n, env.observation_space.n)

# Run LSTD_lambda algorithm:
# print('Running the LSTD Lambda Algorithm ...')
# print("Current Lambda: {0}".format(lambda_))
# LSTD_lambda, theta, loss, G = LSTD_algorithm(trajectories, Phi, num_features, gamma, lambda_)
# print('---------theta------------')
# print("Theta: {0}".format(theta))

# print("#### Compute CV Gradient #####")
# compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D)
# cv_loss = compute_CV_loss(trajectories,Phi, num_features, gamma, lambda_, Gs, logger)
# print("########## Compute CV Loss ###########")
# print("CV Loss: {0}".format(cv_loss))


#cv_loss = compute_CV_loss(trajectories, Phi, num_features, gamma, adaptive_lambda_val, Gs, logger)

#print('Finding optimal lambda using LSTD Lambda Algorithm')
logger = None
if log_events:
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    logger_name = 'Adaptive lambda Algorithm_Gamma_{0:.3}'.format(config.gamma)
    logger = Logger(path + '/temp/logs/train', logger_name)

if config.use_adaptive_lambda:
    print('Running the Adaptive LSTD Lambda Algorithm ...')
    adaptive_LSTD_lambda, adaptive_theta, adaptive_loss, adaptive_G, adaptive_lambda_val = Adaptive_LSTD_algorithm_batch(
                                                                                                                    trajectories, 
                                                                                                                    Phi, 
                                                                                                                    P, 
                                                                                                                    V, 
                                                                                                                    D, 
                                                                                                                    R, 
                                                                                                                    Gs, 
                                                                                                                    logger,
                                                                                                                    config
                                                                                                                    )
    selected_lambda = adaptive_lambda_val
    print('Adaptive Lambda Value: {0}'.format(selected_lambda))
else:
    print('Using default Lambda : {0}'.format(config.default_lambda))
    selected_lambda = config.default_lambda

logger = None
if log_events:
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    logger_name = 'Adaptive lambda_{0:.3}_Gamma_{1:.3}'.format(selected_lambda, config.gamma)
    logger = Logger(path + '/temp/logs/test', logger_name)


cv_loss = compute_CV_loss(trajectories, Phi, config.num_features, config.gamma, selected_lambda, Gs, logger)

#print('Finding optimal lambda using LSTD Lambda Algorithm')
#result = find_adaptive_optimal_lambda_grid_search(trajectories, R, Phi,Gs)
#print('Gamma, Lambda, Loss')
#print(result)
#draw_optimal_lambda_grid_search(gamma=result[:,0], lambda_=result[:,1])
#result = find_optimal_lambda_grid_search(trajectories, Phi,Gs)
# print('Gamma, Lambda, Loss')
# print(result)
# draw_optimal_lambda_grid_search(gamma=result[:,0], lambda_=result[:,1])
# draw_box_grid_search(trajectories, R)
