from lstd_algorithms import minibatch_LSTD, LSTD_algorithm, Adaptive_LSTD_algorithm, Adaptive_LSTD_algorithm_batch, Adaptive_LSTD_algorithm_batch_type2, compute_CV_loss
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_lambda_grid_search(trajectories,P, V, D, R, Phi, Gs, config, logger = False):
    gamma_lambda_loss = []
    gamma = 0.0
    while gamma < 1:
        lambda_ = 0.0
        optimal_loss = np.inf
        while lambda_ < 1:
            print("Finding CV loss for lambda = {0} and gamma = {1}".format(lambda_, gamma))
            #_, _, loss, _ = LSTD_algorithm(trajectories, Phi, num_features, gamma, lambda_)
            _, _, loss, _, _ = Adaptive_LSTD_algorithm_batch(trajectories, Phi, P, V, D, R, Gs, logger, config)
            #loss = compute_CV_loss(trajectories, Phi, num_features, gamma, lambda_, Gs, logger)
            if loss < optimal_loss:
                optimal_loss = loss
                optimal_lambda = lambda_
            print("CV loss for lambda = {0} and gamma = {1} is = {2}".format(lambda_, gamma, loss)) 
            lambda_ += config.step_size_lambda
        gamma_lambda_loss.append([gamma, optimal_lambda, optimal_loss])       
        gamma += config.step_size_gamma
    return np.array(gamma_lambda_loss)

	
def find_adaptive_optimal_lambda_grid_search(trajectories,P, V, D, R, Phi, Gs, config, logger = False):
    gamma_lambda_loss = []
    gamma = 0.0
    while gamma < 1:
        #_, _, optimal_loss, _, optimal_lambda = Adaptive_LSTD_algorithm(trajectories, num_features, Phi, P, V, D, R, lr, gamma, lambda_=0.5, epsilon=0.0)
        _, _, optimal_loss, _, optimal_lambda = Adaptive_LSTD_algorithm_batch(trajectories, Phi, P, V, D, R, Gs, logger, config)
        gamma_lambda_loss.append([gamma, optimal_lambda, optimal_loss])       
        gamma += config.step_size_gamma
    return np.array(gamma_lambda_loss)


def draw_optimal_lambda_grid_search(gamma, lambda_):
    plt.plot(gamma, lambda_, 'ro')
    plt.title('Optimal lambda for each gamma using grid search in 10 iterations')
    plt.ylabel('Optimal lambda')
    plt.xlabel('Gamma')
    plt.grid()
    plt.show()

def set_seed(seed):
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)

'''
Box chart link: http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
'''
def draw_box_grid_search(trajectories, R, initial_seed=1358, seed_iterations=100, seed_step_size=100, step_size_lambda=0.05, step_size_gamma=0.1):
    data = []
    seed = initial_seed
    gamma_length = int(1/step_size_gamma) + 1;
    gammas = [[] for i in range(gamma_length)]

    for i in range(seed_iterations):
        gamma_lambda_loss = find_adaptive_optimal_lambda_grid_search(trajectories, R, Phi, '')
        #gamma_lambda_loss = find_optimal_lambda_grid_search(trajectories, Phi, '')
        #print(gamma_lambda_loss)
        for j in range(gamma_length):
            gammas[j].append(gamma_lambda_loss[j,1])
        
        seed += seed_step_size
        set_seed(seed)
        D, V, trajectories, Gs, R = run_env_episodes(num_episodes)

    for k in range(gamma_length):
        data.append(gammas[k])

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data)
    plt.title('Adaptive lambda for each gamma in 100 iterations')
    #plt.title('Optimal lambda for each gamma using grid search in 100 iterations')
    #gamma range
    ax.set_xticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
    plt.xlabel('Gamma')
    plt.ylabel('Adaptive lambda')
    #plt.ylabel('Optimal lambda')
    plt.show()
    #fig.savefig('box_grid.png', bbox_inches='tight')