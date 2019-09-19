import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from lstd_algorithms import minibatch_LSTD, LSTD_algorithm, Adaptive_LSTD_algorithm, Adaptive_LSTD_algorithm_batch, Adaptive_LSTD_algorithm_batch_type2, compute_CV_loss
from env_utils import init_env, run_env_episodes_walk, run_env_episodes_boyan
import pudb


'''
Use this to find optimal loss and optimal lambda by grid search over lambda for each value of gamma:
'''
def find_optimal_lambda_grid_search(trajectories,P, V, D, R, Phi, Gs, config, logger = False):
    gamma_lambda_loss = []
    gamma = 0.0
    while gamma < 1:
        lambda_ = 0.0
        optimal_loss = np.inf
        while lambda_ < 1:
            print("Finding CV loss for lambda = {0} and gamma = {1}".format(lambda_, gamma))
            new_config = copy.deepcopy(config)
            new_config.gamma = gamma
            new_config.default_lambda = lambda_            
            _, _, loss, _, rmspbe = minibatch_LSTD(trajectories, Phi, P, V, D, R, Gs, logger, new_config)
            #_, _, loss, _, _ = Adaptive_LSTD_algorithm_batch(trajectories, Phi, P, V, D, R, Gs, logger, new_config)
            #loss = compute_CV_loss(trajectories, Phi, num_features, gamma, lambda_, Gs, logger)
            if loss < optimal_loss:
                optimal_loss = loss
                optimal_lambda = lambda_
            print("loss for lambda = {0} and gamma = {1} is = {2}".format(lambda_, gamma, loss)) 
            lambda_ += config.step_size_lambda
        gamma_lambda_loss.append([gamma, optimal_lambda, optimal_loss, rmspbe])       
        gamma += config.step_size_gamma
        if gamma == 1.0 :
            gamma = 0.99
    return np.array(gamma_lambda_loss)
	

'''
Use this to find optimal loss and optimal lambda by maximizing lambda for each value of gamma:
'''
def find_adaptive_optimal_lambda_grid_search(trajectories, 
                                             P, 
                                             V, 
                                             D, 
                                             R, 
                                             Phi, 
                                             Gs, 
                                             config, 
                                             logger = False, 
                                             random_init_lambda = False):
    gamma_lambda_loss = []
    gamma = 0.0
    while gamma < 1:
        new_config = copy.deepcopy(config)
        new_config.gamma = gamma
        if random_init_lambda:
            new_config.default_lambda = random.random()
        _, _, optimal_loss, rmspbe, _, optimal_lambda = Adaptive_LSTD_algorithm_batch(trajectories, Phi, P, V, D, R, Gs, logger, new_config)
        gamma_lambda_loss.append([gamma, optimal_lambda, optimal_loss, rmspbe])    
        gamma += config.step_size_gamma
        if gamma == 1.0 :
            gamma = 0.99
    return np.array(gamma_lambda_loss)

def draw_optimal_lambda_grid_search(gamma, lambda_, file_path, config, ylabel):
    plt.figure()
    plt.plot(gamma, lambda_, 'ro')
    plt.title('Optimal lambda for each gamma using grid search in {0} iterations'.format(config.num_episodes))
    plt.ylabel(ylabel)
    plt.xlabel('Gamma')
    plt.grid()
    plt.savefig(file_path)
    plt.close()
    
def set_seed(seed, env):
    env.reset()
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)

'''
Box chart link: http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
Use this only for adaptive lambda
'''
def draw_box_grid_search_adaptive_lambda(env,
                         P,
                         Phi, 
                         config, 
                         logger,
                         seed_iterations=100, 
                         seed_step_size=100, 
                         step_size_lambda=0.05, 
                         step_size_gamma=0.1,
                         file_paths = ['box_graph_lambda', 'box_graph_loss', 'box_graph_rmspbe'],
                         random_init_lambda = False):
    lambda_data = []
    loss_data = []
    rmspbe_data = []
    seed = config.seed
    gamma_length = int(1/step_size_gamma) + 1;
    lambdas = [[] for i in range(gamma_length)]
    losses = [[] for i in range(gamma_length)]
    rmspbes = [[] for i in range(gamma_length)]
    for i in range(seed_iterations):
        set_seed(seed, env)
        
    if 'walk' in config.env_name:
        D, V, trajectories, Gs, R = run_env_episodes_walk(env, config)

    else:
        D, V, trajectories, Gs, R = run_env_episodes_boyan(env, config)
    
        gamma_lambda_loss = find_adaptive_optimal_lambda_grid_search(trajectories,
                                                                     P, 
                                                                     V, 
                                                                     D, 
                                                                     R, 
                                                                     Phi, 
                                                                     Gs, 
                                                                     config, 
                                                                     logger,
                                                                     random_init_lambda = random_init_lambda)
        for j in range(gamma_length):
            lambdas[j].append(gamma_lambda_loss[j,1])
            losses[j].append(gamma_lambda_loss[j,2])
            rmspbes[j].append(gamma_lambda_loss[j,3])
        seed += seed_step_size

    for k in range(gamma_length):
        lambda_data.append(lambdas[k])
        loss_data.append(losses[k])
        rmspbe_data.append(rmspbes[k])
    ###Plot lambda box plot:
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(lambda_data)
    plt.title('Adaptive lambda for each gamma in for {0} episodes, initial lambda:{1}'.format(config.num_episodes, config.default_lambda))
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('Adaptive lambda')
    fig.savefig(file_paths[0], bbox_inches='tight')
    plt.close()
    ###Plot loss box plot:
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(loss_data)
    plt.title('Adaptive lambda loss for each gamma in for {0} episodes, initial lambda:{1}'.format(config.num_episodes, config.default_lambda))
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('Loss')
    fig.savefig(file_paths[1], bbox_inches='tight')
    plt.close()

    ###Plot RMSPBE box plot:
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(rmspbe_data)
    plt.title('Optimal lambda RMSPBE for each gamma in for {0} episodes'.format(config.num_episodes))
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('RMSPBE')
    fig.savefig(file_paths[2], bbox_inches='tight')
    plt.close()
    
    
'''
Box chart link: http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
Use this only for optimal lambda
'''
def draw_box_grid_search_optimal_lambda(env,
                         P,
                         Phi, 
                         config, 
                         logger,
                         seed_iterations=100, 
                         seed_step_size=100, 
                         step_size_lambda=0.05, 
                         step_size_gamma=0.1,
                         file_paths = ['box_graph_lambda', 'box_graph_loss', 'box_graph_rmspbe'],
                         random_init_lambda = False):
    lambda_data = []
    loss_data = []
    rmspbe_data = []
    seed = config.seed
    gamma_length = int(1/step_size_gamma) + 1;
    lambdas = [[] for i in range(gamma_length)]
    losses = [[] for i in range(gamma_length)]
    rmspbes = [[] for i in range(gamma_length)]
    
    for i in range(seed_iterations):
        set_seed(seed, env)
        if 'walk' in config.env_name:
            D, V, trajectories, Gs, R = run_env_episodes_walk(env, config)

        else:
            D, V, trajectories, Gs, R = run_env_episodes_boyan(env, config)
            gamma_lambda_loss = find_optimal_lambda_grid_search(trajectories,P, V, D, R, Phi, Gs, config)
        for j in range(gamma_length):
            lambdas[j].append(gamma_lambda_loss[j,1])
            losses[j].append(gamma_lambda_loss[j,2])
            rmspbes[j].append(gamma_lambda_loss[j,3])
        seed += seed_step_size

    for k in range(gamma_length):
        lambda_data.append(lambdas[k])
        loss_data.append(losses[k])
        rmspbe_data.append(rmspbes[k])
   
    ###Plot lambda box plot:
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(lambda_data)
    plt.title('Optimal lambda for each gamma in for {0} episodes'.format(config.num_episodes))
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('Optimal lambda')
    fig.savefig(file_paths[0], bbox_inches='tight')
    plt.close()
    ###Plot loss box plot:
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(loss_data)
    plt.title('Optimal lambda loss for each gamma in for {0} episodes'.format(config.num_episodes))
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('Loss')
    fig.savefig(file_paths[1], bbox_inches='tight')
    plt.close()

    ###Plot RMSPBE box plot:
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(rmspbe_data)
    plt.title('Optimal lambda RMSPBE for each gamma in for {0} episodes'.format(config.num_episodes))
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('RMSPBE')
    fig.savefig(file_paths[2], bbox_inches='tight')
    plt.close()