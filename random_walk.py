#import gym_walk
from boyan_exp import BOYAN_MDP

import pdb
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from pprint import pprint
from grid_search_utils import find_optimal_lambda_grid_search, find_adaptive_optimal_lambda_grid_search, draw_optimal_lambda_grid_search
from grid_search_utils import draw_box_grid_search_adaptive_lambda, draw_box_grid_search_optimal_lambda
from lstd_algorithms import minibatch_LSTD, LSTD_algorithm, Adaptive_LSTD_algorithm, Adaptive_LSTD_algorithm_batch, Adaptive_LSTD_algorithm_batch_type2, compute_CV_loss, Adaptive_LSTD_algorithm_batch_type3, minibatch_LSTD_withCV
from compute_utils import get_discounted_return, compute_P
from env_utils import init_env, run_env_episodes_boyan, run_env_episodes_walk, run_env_episodes_taxi, taxi_env_features
from Config import Config
from pprint import pprint
import pudb
import pickle
import pathlib


################# Input Arguments ################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1358)
parser.add_argument('--lr', type=float, default= 0.5)
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--default_lambda', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=.8)
parser.add_argument('--rand_lambda', type=bool, default=False)
parser.add_argument('--walk_type', type=str, default='tabular')

args = parser.parse_args()

##########################################################

################  Parameters #################
done = False
log_events = True

if log_events:
    from tensorboard_utils import Logger
        
#if mdp: num_features = 4, num_states = 13.
#if randomwwalk: tabular, inverted: num_features = 5, num_states = 5, dependent = 3,5
config = Config(
    seed = args.seed,
    #env_name = 'RandomWalkFive-v0',
    #env_name = 'Boyan',
    env_name = 'Taxi_3',
    walk_type = args.walk_type,
    num_features = 7,#4,
    num_states = 7,#13,
    num_train_episodes = args.episodes,
    num_test_episodes = 50,
    A_inv_epsilon = 1e-3,
    gamma = args.gamma,
    default_lambda = args.default_lambda,
    lr = args.lr,
    use_adaptive_lambda = True,
    grad_clip_norm = 10,
    compute_autograd = False,
    use_adam_optimizer = True,
    batch_size = args.batch,
    upsampling_rate = 1,
    step_size_gamma = 0.1,
    step_size_lambda = 0.02,
    seed_iterations=5, 
    seed_step_size=5, 
    random_init_lambda = args.rand_lambda,
    rcond = 1e-14,
    compute_cv_iterations = 10,
    taxi_grid_size = 3,
)

##########################################################
run_id = random.randint(0,10000)
print('run_id:{0}'.format(run_id))
print(config)
env = init_env(config.env_name, config.seed)

if 'Walk' in config.env_name:
    transition_probs = env.env.P
    D, V, trajectories, Gs, R = run_env_episodes_walk(env, config, 'train')
    D_test, V_test, trajectories_test, Gs_test, R_test = run_env_episodes_walk(env, config, 'test')
    
elif 'Boyan' in config.env_name:
    transition_probs = env.transitions
    D, V, trajectories, Gs, R = run_env_episodes_boyan(env, config, 'train')
    D_test, V_test, trajectories_test, Gs_test, R_test = run_env_episodes_boyan(env, config, 'test')
elif 'Taxi_5' in config.env_name:
    transition_probs = env.P
    file = pathlib.Path("train_D_V_traj_Gs_R.pkl")
    if file.exists():
        with open('train_D_V_traj_Gs_R.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            D, V, trajectories, Gs, R = pickle.load(f)
    else:
        D, V, trajectories, Gs, R = run_env_episodes_taxi(env, config, 'train')
        with open('train_D_V_traj_Gs_R.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([D, V, trajectories, Gs, R], f)  
    file = pathlib.Path("test_D_V_traj_Gs_R.pkl")
    if file.exists():
        with open('test_D_V_traj_Gs_R.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            D_test, V_test, trajectories_test, Gs_test, R_test = pickle.load(f)
    else:
        D_test, V_test, trajectories_test, Gs_test, R_test = run_env_episodes_taxi(env, config, 'test')
        with open('test_D_V_traj_Gs_R.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([D_test, V_test, trajectories_test, Gs_test, R_test], f) 
elif 'Taxi_3' in config.env_name:
    transition_probs = env.env.P
    file = pathlib.Path("small_train_D_V_traj_Gs_R.pkl")
    if file.exists():
        with open('small_train_D_V_traj_Gs_R.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            D, V, trajectories, Gs, R = pickle.load(f)
    else:
        D, V, trajectories, Gs, R = run_env_episodes_taxi(env, config, 'train')
        with open('small_train_D_V_traj_Gs_R.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([D, V, trajectories, Gs, R], f)  
    file = pathlib.Path("small_test_D_V_traj_Gs_R.pkl")
    if file.exists():
        with open('small_test_D_V_traj_Gs_R.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            D_test, V_test, trajectories_test, Gs_test, R_test = pickle.load(f)
    else:
        D_test, V_test, trajectories_test, Gs_test, R_test = run_env_episodes_taxi(env, config, 'test')
        with open('small_test_D_V_traj_Gs_R.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([D_test, V_test, trajectories_test, Gs_test, R_test], f) 
print("###############Transition Probabilities####################")
print(transition_probs)
##Upsample 1's:
#upsampled_Gs, upsampled_trajectories = upsample_trajectories(Gs, trajectories, config.upsampling_rate)

if 'Walk' in config.env_name:
    if config.walk_type == 'tabular':      
        Phi = np.array([[1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,1,0,0],
                        [0,0,0,1,0],
                        [0,0,0,0,1]])
        config.num_states = 5
        config.num_features = 5
    elif config.walk_type == 'inverted':      
        Phi = np.array([[0,0.5,0.5,0.5,0.5],
                        [0.5,0,0.5,0.5,0.5],
                        [0.5,0.5,0,0.5,0.5],
                        [0.5,0.5,0.5,0,0.5],
                        [0.5,0.5,0.5,0.5,0]])
        config.num_features = 5
        config.num_states = 5
        
    elif config.walk_type == 'dependent':      
        Phi = np.array([[1,0,0],
                        [1/2**0.5, 1/2**0.5, 0],
                        [1/3**0.5, 1/3**0.5, 1/3**0.5],
                        [0, 1/2**0.5, 1/2**0.5],
                        [0,0,1]])
        config.num_features = 3
        config.num_states = 5

    else:
        Phi = np.array(np.random.rand(config.num_states, config.num_features))
elif 'Boyan' in config.env_name:
    Phi= 1/4 * np.array([[4, 0,0,0],[3,1,0,0],[2,2,0,0],[1,3,0,0],[0,4,0,0],[0,3,1,0], [0,2,2,0], [0,1,3,0], [0,0,4,0],
                        [0,0,3,1], [0,0,2,2], [0,0,1,3], [0,0,0,4]])
    config.num_features = 4
    config.num_states = 13  
elif '2048' in config.env_name:
    config.num_features = 16
    config.num_states = 15**16
    Phi = np.array(np.random.rand(config.num_states, config.num_features))
elif 'Taxi' in config.env_name:
    config.num_features = config.taxi_grid_size ** 2
    config.num_states = 20 * config.taxi_grid_size ** 2
    Phi = taxi_env_features(transition_probs, config.taxi_grid_size)

'''
Now compute the MRP value of P: P(s'|s)
'''

if 'Walk' in config.env_name or 'Taxi' in config.env_name:
    P = compute_P(transition_probs, env.action_space.n, env.observation_space.n)
else:
    transitions = transition_probs
    boyan_mdp = BOYAN_MDP('boyan_mdp.png')
    mdp_spec = boyan_mdp.spec
    P = np.zeros(shape=(env.observation_space.n, env.observation_space.n))
    expected_rewards = np.zeros(shape=(env.observation_space.n, env.action_space.n,))
    for (state, action), choices in transitions.next_states.items():
        for next_state, prob in choices.items():
            P[state.index, next_state.index] = prob
    for state in mdp_spec.states:
        if state.terminal_state:
            P[state.index, state.index] = 1.
    P /= P.sum(axis=1)[:,np.newaxis]   
    for (state, action), choices in transitions.rewards.items():
            expected_rewards[state.index, action.index] = sum(value * prob for value, prob in choices.items())

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
    adaptive_LSTD_lambda, adaptive_theta, adaptive_avg_losses, adaptive_G, adaptive_lambda_val = Adaptive_LSTD_algorithm_batch_type3(
                                                                                                                    trajectories, 
                                                                                                                    Phi, 
                                                                                                                    P, 
                                                                                                                    V, 
                                                                                                                    D, 
                                                                                                                    R, 
                                                                                                                    Gs, 
                                                                                                                    logger,
                                                                                                                    config,
                                                                                                                    trajectories_test,
                                                                                                                    Gs_test
                                                                                                                    )
    selected_lambda = adaptive_lambda_val
    print('Adaptive Lambda Value: {0}'.format(selected_lambda))
else:
    print('Using default Lambda : {0}'.format(config.default_lambda))
    #pudb.set_trace()
    new_config = copy.deepcopy(config)
    new_config.default_lambda = 0
    adaptive_LSTD_lambda, adaptive_theta, adaptive_G, adaptive_ms_loss, adaptive_rms_loss = minibatch_LSTD_withCV(trajectories, 
                                                                                                                    Phi, 
                                                                                                                    P, 
                                                                                                                    V, 
                                                                                                                    D, 
                                                                                                                    R, 
                                                                                                                    Gs,
                                                                                                                    logger, 
                                                                                                                    new_config,
                                                                                                                    trajectories_test,
                                                                                                                    Gs_test
                                                                                                                    )
print(adaptive_theta)
selected_lambda = config.default_lambda
logger = None
if log_events:
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    logger_name = 'Adaptive lambda_{0:.3}_Gamma_{1:.3}'.format(selected_lambda, config.gamma)
    logger = Logger(path + '/temp/logs/test', logger_name)

#cv_loss = compute_CV_loss(trajectories, Phi, config.num_features, config.gamma, selected_lambda, Gs, logger)

config_prefix = config.to_str()
print(config_prefix)
########## run adaptive lambda over all values of gamma and lambda and plot the graph:
###Uncomment below if need to run over a specific value of lambda:
# config['default_lambda'] = 0.5
# result = find_adaptive_optimal_lambda_grid_search(trajectories,P, V, D, R, Phi, Gs, config)
# print('Gamma, Lambda, Loss')
# print(result)
# dirpath = os.getcwd()
# file_name = 'adaptive_lambda_lambda_{1}_lr_{2}_gridsearch.png'.format(config.gamma, config.default_lambda, config.lr)
# fig_file_name = os.path.join(dirpath, 'figures',file_name)
# draw_optimal_lambda_grid_search(gamma=result[:,0], lambda_=result[:,1], file_path = fig_file_name)

#Use below to draw the box plot for adaptive lambda algorithm: The only variables are learning rate and number of episodes:
# dirpath = os.getcwd()
# file_names = ['adaptive_lambda_lambdas_box_graph_{0}_gridsearch_runid_{1}.png'.format(config_prefix, run_id),
#               'adaptive_lambda_loss_box_graph_{0}_gridsearch_runid_{1}.png'.format(config_prefix, run_id)]
# fig_file_names = [os.path.join(dirpath, 'figures',file_names[0]),
#                   os.path.join(dirpath, 'figures',file_names[1])]
# draw_box_grid_search_adaptive_lambda(env,
#                      P,
#                      Phi, 
#                      config, 
#                      logger,
#                      seed_iterations=config.seed_iterations, 
#                      seed_step_size=config.seed_step_size, 
#                      step_size_lambda=config.step_size_lambda, 
#                      step_size_gamma=config.step_size_gamma,
#                      file_paths = fig_file_names,
#                      random_init_lambda = config.random_init_lambda
#                      )
########## Find optimal lambda for each gamma. This is using lstd algorithm, not searching for lambda:
# dirpath = os.getcwd()
# file_names = ['optimal_lambda_lstd_lambda_lambdas_{0}_gridsearch_runid_{1}.png'.format(config_prefix, run_id),
#               'optimal_lambda_lstd_lambda_losses_{0}_gridsearch_runid_{1}.png'.format(config_prefix, run_id)]
# fig_file_names = [os.path.join(dirpath, 'figures',file_names[0]),
#                   os.path.join(dirpath, 'figures',file_names[1])]
# draw_box_grid_search_optimal_lambda(env,
#                      P,
#                      Phi, 
#                      config, 
#                      logger,
#                      seed_iterations=config.seed_iterations, 
#                      seed_step_size=config.seed_step_size, 
#                      step_size_lambda=config.step_size_lambda, 
#                      step_size_gamma=config.step_size_gamma,
#                      file_paths = fig_file_names,
#                      random_init_lambda = config.random_init_lambda
#                      )

