#import gym_walk
from boyan_exp import BOYAN_MDP

import pdb
import numpy as np
import random
import matplotlib.pyplot as plt
from pprint import pprint
from grid_search_utils import find_optimal_lambda_grid_search, find_adaptive_optimal_lambda_grid_search, draw_optimal_lambda_grid_search, draw_box_grid_search
from lstd_algorithms import minibatch_LSTD, LSTD_algorithm, Adaptive_LSTD_algorithm, Adaptive_LSTD_algorithm_batch, Adaptive_LSTD_algorithm_batch_type2, compute_CV_loss
from compute_utils import get_discounted_return, compute_P
from env_utils import init_env, run_env_episodes
from Config import Config
import pudb


################# Input Arguments ################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1358)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--default_lambda', type=float, default=0.75)
parser.add_argument('--gamma', type=float, default=1.0)
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
    #env_name = 'RandomWalk-v0',
    env_name = 'Boyan',
    walk_type = args.walk_type,
    num_features = 5,#4,
    num_states = 5,#13,
    num_episodes = args.episodes,
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
)

##########################################################
run_id = random.randint(0,10000)
print('run_id:{0}'.format(run_id))
print(config)
env = init_env(config.env_name, config.seed)
if config.env_name == 'RandomWalk-v0':
    transition_probs = env.env.P
else:
    config.num_features = 4
    config.num_states = 13
    transition_probs = env.transitions
print("###############Transition Probabilities####################")
print(transition_probs)
print('Generate Monte Carlo Estimates of D and V...')
D, V, trajectories, Gs, R = run_env_episodes(env, config)
print('Done finding D and V!')
##Upsample 1's:
#upsampled_Gs, upsampled_trajectories = upsample_trajectories(Gs, trajectories, config.upsampling_rate)
if config.env_name == 'RandomWalk-v0':
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
        Phi = np.array(np.random(config.num_states, config.num_features))
else:
    Phi= 1/4 * np.array([[4, 0,0,0],[3,1,0,0],[2,2,0,0],[1,3,0,0],[0,4,0,0],[0,3,1,0], [0,2,2,0], [0,1,3,0], [0,0,4,0],
                        [0,0,3,1], [0,0,2,2], [0,0,1,3], [0,0,0,4]])
    config.num_features = 4
    config.num_states = 13  

'''
Now compute the MRP value of P: P(s'|s)
'''

if config.env_name == 'RandomWalk-v0':
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
    #pdb.set_trace()
    adaptive_LSTD_lambda, adaptive_theta, adaptive_loss, adaptive_G= minibatch_LSTD(trajectories, Phi, config.num_features, config.gamma, selected_lambda)
#pdb.set_trace()
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
# draw_box_grid_search(env,
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
result = find_optimal_lambda_grid_search(trajectories,P, V, D, R, Phi, Gs, config)
dirpath = os.getcwd()
file_names = ['optimal_lambda_lstd_lambda_lambdas_{0}_gridsearch_runid_{1}.png'.format(config_prefix, run_id),
              'optimal_lambda_lstd_lambda_losses_{0}_gridsearch_runid_{1}.png'.format(config_prefix, run_id)]
fig_file_names = [os.path.join(dirpath, 'figures',file_names[0]),
                  os.path.join(dirpath, 'figures',file_names[1])]
draw_optimal_lambda_grid_search(gamma=result[:,0], lambda_=result[:,1], file_path = fig_file_names[0], config=config, ylabel='lambda')
draw_optimal_lambda_grid_search(gamma=result[:,0], lambda_=result[:,2], file_path = fig_file_names[1], config=config, ylabel='loss')
