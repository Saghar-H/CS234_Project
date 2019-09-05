import gym
import gym_walk
import pdb
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import utils
from lstd import LSTD
from pprint import pprint
from adam import ADAM
from autograd_cls import AutoGrad 
 
################  Parameters #################
done = False
seed = 1358
env_name = 'WalkFiveStates-v0'
num_features = 10
num_states = 5
num_episodes = 10000

gamma = 0.9
default_lambda = 0.5
lr = 0.0001
log_events = True
use_adaptive_lambda = True
compute_autograd = False
if log_events:
    from tensorboard_utils import Logger
   
##########################################################

def init_env(env_name, seed):
    env = gym.make(env_name)
    env.reset()
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    return env


def calculate_batch_loss(trajectories, G, theta):
    '''
    :param theta: Value function parameter such that V= Phi * Theta
    :param trajectories: dictionary of episodes trajectories: {ep0 : [(state, reward, state_next, done), ...]}
    :param G: dictionary of episodes return values: {ep0 : [g0, g1, ... ]}
    :return: list of episodes loss, average over episodes's loss
    '''
    num_episodes = len(trajectories.keys())
    loss = []
    for ep in range(num_episodes):
        traj = trajectories[ep]
        if len(traj) <= 4 or len(G[ep]) <= 0:
            continue
        ep_loss = np.mean(
            [(np.dot(Phi[traj[t][0], :], theta) - G[ep][t]) ** 2 for t in range(len(traj))])
        loss.append(ep_loss)
    avg_loss = np.mean(loss)
    return loss, avg_loss


def get_discounted_return(episode_rewards, gamma):
    discounted_rewards = [0] * (len(episode_rewards) + 1)
    for i in range(len(episode_rewards) - 1, -1, -1):
        discounted_rewards[i] = discounted_rewards[i + 1] * gamma + episode_rewards[i]
    return discounted_rewards[:-1]


def compute_P(transition_probs, num_actions, num_states):
    ret = np.zeros((num_states, num_states))
    for s in transition_probs.keys():
        for a in transition_probs[s]:
            for tup in transition_probs[s][a]:
                sp = tup[1]
                p_sasp = tup[0]
                ret[s, sp] += 1.0 / num_actions * p_sasp
    return ret


def compute_cv_gradient(phi, theta, gamma, lstd_lambda, P, V, D, R):
    #print('****rewards*****')
    #print(V)
    #print(R)
    I = np.eye(len(P), len(P[0]))
    phi_t = phi.transpose()
    #V_t = V.transpose()
    R_t = R.transpose()
    I_gamma_P = I - gamma * P
    # print('*****---- P ----*****')
    # print(P)
    inv1 = np.linalg.pinv(I - gamma * lstd_lambda * P)
    #psi = phi_t @ D @ inv1 @ I_gamma_P
    psi = phi_t @ D @ inv1
    # print('*****---- psi ----*****')
    # print(psi)
    inv2 = np.linalg.pinv(psi @ phi)
    H = phi @ inv2 @ psi
    I_H = I - H
    d = np.diag(np.diag(I_H))
    # print('*****----H----*****')
    # print(H)
    # print('*****----diag(I-H)----*****')
    # print(d)
    d_inv1 = np.linalg.pinv(d)
    # print('*****----(diag(I-H))^(-1)----*****')
    # print(d_inv1)
    d_inv2 = np.linalg.pinv(d_inv1)
    # print('*****----(diag(I-H))^(-2)----*****')
    # print(d_inv2)
    psi_gradient = gamma * phi_t @ D @ inv1 @ P @ inv1 @ I_gamma_P
    H_gradient = phi @ inv2 @ (psi_gradient - psi_gradient @ phi @ inv2 @ psi)
    diag_H_gradient = np.diag(np.diag(H_gradient))
    d_inv2_gradient = d_inv1 @ (diag_H_gradient @ d_inv1 + d_inv1 @ diag_H_gradient) @ d_inv1
    #d_inv2_gradient = 2 * d_inv1 @ diag_H_gradient @ d_inv2
    #print('-------')
    #print(d_inv1 @ (diag_H_gradient @ d_inv1 + d_inv1 @ diag_H_gradient) @ d_inv1)
    #print(d_inv2_gradient)
    # print('*****---- H_gradient ----*****')
    # print(H_gradient)
    #H_V = phi @ theta
    term1 = -R_t @ H_gradient @ d_inv2 @ I_H @ R
    term2 = R_t @ I_H @ d_inv2_gradient @ I_H @ R
    term3 = -R_t @ I_H @ d_inv2 @ H_gradient @ R
    cv_gradient = term1 + term2 + term3
    # print("#### CV Gradient #####")
    # print(cv_gradient)
    return cv_gradient

#def compute_cv_gradient2(Phi, theta, gamma, lambda_, R, A, b, z)
#    A_inv_lambda_gradient = utils.compute_A_inv_gradient(A, b, z, Phi)
#    b_lambda_gradient = utils.compute_b_grandient(b)
#    z_lambda_gradient = compute_z_gradient()
#    hjj_lambda_gradient = compute_hjj_gradient()
#    epsilon_lambda_gradient = compute_epsilon_lambda_gradient()
#    lcv_h_gradient = compute_lcv_h_gradient()

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

        ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
        Gs[ep] = ep_discountedrewards

        for i in range(len(ep_states)):
            V[ep_states[i]] += ep_discountedrewards[i]
            R[ep_states[i]] += ep_rewards[i]

    #print('Monte Carlo D:{0}'.format(D * 1.0 / total_steps, total_steps))
    #print('Monte Carlo V:{0}'.format(V * 1.0 / D))
    #print('----------Trajectories---------')
    #print(trajectories[0])
    return np.diag(D / total_steps), V / D, trajectories, Gs, R/D


def LSTD_algorithm(trajectories, Phi, num_features, gamma=0.4, lambda_=0.2, epsilon=0.0):
    # LSTD operator:
    LSTD_lambda = LSTD(num_features, epsilon=0.0)
    G = {}
    running_loss = []
    num_episodes = len(trajectories.keys())
    for ep in range(num_episodes):
        G[ep] = []
        traj = trajectories[ep]
        if len(traj) <= 4:
            continue
        ep_rewards = []
        ep_states = []
        cur_state = traj[0][0]
        LSTD_lambda.reset_boyan(Phi[cur_state, :])
        for timestep in range(len(traj)):
            cur_state, reward, next_state, done = traj[timestep]
            LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        theta = LSTD_lambda.theta
        ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
        # print('ep_discounted:{0}'.format(ep_discountedrewards))
        if len(ep_discountedrewards) > 0:
            ep_loss = np.mean(
                [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])
            # print('Episode {0} loss is {1}'.format(ep, ep_loss))
            # print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
            G[ep] = ep_discountedrewards
            running_loss.append(ep_loss)
    # After we calculated the Theta parameter from the training data
    loss, _ = calculate_batch_loss(trajectories, G, theta)
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)

    # print("average running loss in training: ", sum(running_loss) / num_episodes)
    # print("average loss after training: ", sum(loss) / num_episodes)
    average_loss = np.mean(loss)
    return LSTD_lambda, theta, average_loss, G


def Adaptive_LSTD_algorithm(trajectories, num_features, Phi, P, V, D, R, Gs, lr=0.1, gamma=0.4, lambda_=0.2, epsilon=0.0):
    # LSTD operator:
    Auto_grad = AutoGrad(compute_CV_loss, 4)
    Auto_grad.gradient_fun()
    adaptive_LSTD_lambda = LSTD(num_features, epsilon=0.0)
    G = {}
    loss = []
    running_loss = []
    num_episodes = len(trajectories.keys())
    adam_optimizer = ADAM()
    for ep in range(num_episodes):
        G[ep] = []
        traj = trajectories[ep]
        if len(traj) <= 4:
            continue
        ep_rewards = []
        ep_states = []
        Z = np.zeros((num_features, num_states))
        Z_gradient = np.zeros((num_features, num_states))
        H_diag = np.zeros(num_states) # n 
        eps = np.zeros(num_states)
        states_count = np.zeros(num_states)
        epsilon_lambda_gradient = np.zeros(num_states)
        H_diag_gradient = np.zeros(num_states)
        episode_loss = 0
        cur_state = traj[0][0]
        adaptive_LSTD_lambda.reset_boyan(Phi[cur_state, :])

        for timestep in range(len(traj)):

            cur_state, reward, next_state, done = traj[timestep]
            adaptive_LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        theta = adaptive_LSTD_lambda.theta
        A = adaptive_LSTD_lambda.A
        b = adaptive_LSTD_lambda.b
        A_inv = np.linalg.pinv(A)
        for timestep in range(len(traj)-1):
            cur_state, reward, next_state, done = traj[timestep]
            # To-do : change the following update to running average
            states_count[cur_state] += 1
            ct = states_count[cur_state]
            Z[:,cur_state] = (ct-1)/ct *Z[:,cur_state]+ 1/ct * utils.compute_z(lambda_, gamma, Phi, ep_states, timestep )
            Z_gradient[:, cur_state] = (ct-1)/ct * Z_gradient[:, cur_state] + 1/ct * utils.compute_z_gradient(lambda_, gamma, Phi, ep_states, timestep)            
            H_diag[cur_state] = (ct-1)/ct * H_diag[cur_state] + 1/ct * utils.compute_hjj(Phi, lambda_, gamma, ep_states, timestep, A_inv)
            eps[cur_state] = (ct-1)/ct * eps[cur_state] + 1/ct * utils.compute_eps_t(Phi, theta, gamma, reward, ep_states, timestep)
            epsilon_lambda_gradient[cur_state] = (ct-1)/ct * epsilon_lambda_gradient[cur_state] + 1/ct * utils.compute_epsilon_lambda_gradient(Phi, lambda_, gamma,A, b,  A_inv, Z, timestep, ep_states, ep_rewards)
            H_diag_gradient[cur_state] = (ct-1)/ct * H_diag_gradient[cur_state] + 1/ct * utils.compute_hjj_gradient(Phi, lambda_, gamma, ep_states, timestep, A, b,  A_inv)
        #grad = compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D, R)
        # Replaced the above update with:
        grad = utils.compute_lcv_lambda_gradient(eps, H_diag, ep_states, epsilon_lambda_gradient, H_diag_gradient)
        if compute_autograd:    
            auto_grad = Auto_grad.loss_autograd_fun(trajectories, Phi, num_features, gamma, lambda_, Gs)
            print('gradient diff:{0}'.format(abs(grad-auto_grad)))

        # if ep > 1000 :
        # new_lambda = lambda_ -  lr * compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D)
        # print(new_lambda)
        # if new_lambda >= 0 and new_lambda <= 1:
        #   lambda_ = new_lambda
        #   print('current lambda:{0}'.format(lambda_))
        
        # grad = compute_cv_gradient2(Phi, theta, gamma, lambda_, R, A, b, z)
        # adam_optimizer.update(grad, ep)
        # new_lambda = adam_optimizer.theta
        new_lambda = lambda_ - lr * grad
        if new_lambda >= 0 and new_lambda <= 1:
            lambda_ = new_lambda
            print('current lambda:{0}'.format(lambda_))
        ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
        # print('ep_discounted:{0}'.format(ep_discountedrewards))
        if len(ep_discountedrewards) > 0:
            ep_loss = np.mean(
                [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in range(len(ep_states))])
            # print('Episode {0} loss is {1}'.format(ep, ep_loss))
            # print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
            G[ep] = ep_discountedrewards
            running_loss.append(ep_loss)
    # After we calculated the Theta parameter from the training data
    loss, _ = calculate_batch_loss(trajectories, G, theta)
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)
    #print("Final Lambda: {0}".format(lambda_))
    #print("average running loss in training: ", np.mean(running_loss))
    #print("average loss after training: ", np.mean(loss))
    return adaptive_LSTD_lambda, theta, np.mean(loss), G, lambda_


def compute_CV_loss(trajectories, Phi, num_features, gamma, lambda_, Gs, logger=False,
                    epsilon=0.0):
    '''
    :param trajectories:
    :param num_features:
    :param gamma:
    :param epsilon:
    :return:
    '''
    total_num_tuples = sum([len(traj) for traj in trajectories.values()])
    num_episodes = len(trajectories.keys())
    loto_loss = []
    step = 0
    for i in range(min(1000,num_episodes)):
        traj = trajectories[i]
        if len(traj) <= 4:
            continue
        for j in range(len(traj)):
            # leave one tuple oto_trajectoriesout
            loto_trajectories = copy.deepcopy(trajectories)
            del loto_trajectories[i][j]
            model, _, loss, _ = LSTD_algorithm(loto_trajectories, Phi, num_features, gamma, lambda_)
            theta = model.theta
            # pdb.set_trace()
            tuple_loss = (np.dot(Phi[trajectories[i][j][0], :], theta) - Gs[i][j]) ** 2
            loto_loss.append(tuple_loss)
            
            if logger:
                logger.log_scalar('average trajectories loss', loss, step)
                logger.log_scalar('current tuple loto cv', tuple_loss, step)
                logger.log_scalar('mean loto cv', np.mean(loto_loss), step)
                logger.writer.flush()
                step += 1

                print('trajectory :{0}, current mean loto loss:{1}'.format(i, np.mean(loto_loss)))

    cv_loss = np.mean(loto_loss)
    return cv_loss


def find_optimal_lambda_grid_search(trajectories, Phi, Gs, step_size_lambda=0.05, step_size_gamma=0.1, logger = False):
    gamma_lambda_loss = []
    gamma = 0.0
    while gamma < 1:
        lambda_ = 0.0
        optimal_loss = np.inf
        while lambda_ < 1:
            print("Finding CV loss for lambda = {0} and gamma = {1}".format(lambda_, gamma))
            _, _, loss, _ = LSTD_algorithm(trajectories, Phi, num_features, gamma, lambda_)
            #loss = compute_CV_loss(trajectories, Phi, num_features, gamma, lambda_, Gs, logger)
            if loss < optimal_loss:
                optimal_loss = loss
                optimal_lambda = lambda_
            print("CV loss for lambda = {0} and gamma = {1} is = {2}".format(lambda_, gamma, loss)) 
            lambda_ += step_size_lambda
        gamma_lambda_loss.append([gamma, optimal_lambda, optimal_loss])       
        gamma += step_size_gamma
    return np.array(gamma_lambda_loss)

	
def find_adaptive_optimal_lambda_grid_search(trajectories, R, Phi, Gs, step_size_lambda=0.05, step_size_gamma=0.1, logger = False):
    gamma_lambda_loss = []
    gamma = 0.0
    while gamma < 1:
        #_, _, optimal_loss, _, optimal_lambda = Adaptive_LSTD_algorithm(trajectories, num_features, Phi, P, V, D, R, lr, gamma, lambda_=0.5, epsilon=0.0)
        _, _, optimal_loss, _, optimal_lambda = Adaptive_LSTD_algorithm(trajectories, num_features, Phi, P, V, D, R, Gs, lr, lambda_=np.random.rand(), epsilon=0.0)
        gamma_lambda_loss.append([gamma, optimal_lambda, optimal_loss])       
        gamma += step_size_gamma
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


env = init_env(env_name, seed)

transition_probs = env.env.P
print("###############Transition Probabilities####################")
print(transition_probs)
print('Generate Monte Carlo Estimates of D and V...')
D, V, trajectories, Gs, R = run_env_episodes(num_episodes)
print('Done finding D and V!')
Phi = np.random.rand(num_states, num_features)
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

if use_adaptive_lambda:
    print('Running the Adaptive LSTD Lambda Algorithm ...')
    adaptive_LSTD_lambda, adaptive_theta, adaptive_loss, adaptive_G, adaptive_lambda_val = Adaptive_LSTD_algorithm(trajectories, num_features,
                                                                                              Phi, P, V, D, R, Gs, lr,
                                                                                              gamma, default_lambda)
    selected_lambda = adaptive_lambda_val
    print('Adaptive Lambda Value: {0}'.format(selected_lambda))
else:
    print('Using default Lambda : {0}'.format(default_lambda))
    selected_lambda = default_lambda

# print('Running the Adaptive LSTD Lambda Algorithm ...')
# adaptive_LSTD_lambda, adaptive_theta, adaptive_loss, adaptive_G, adaptive_lambda_val = Adaptive_LSTD_algorithm(trajectories, num_features,
#                                                                                           Phi, P, V, D, R, Gs, lr,
#                                                                                          gamma, default_lambda)

logger = None
if log_events:
    logger_name = 'Adaptive lambda_{0:.3}_Gamma_{1:.3}'.format(selected_lambda, gamma)
    logger = Logger('/Users/siamak/temp/logs/test', logger_name)


cv_loss = compute_CV_loss(trajectories, Phi, num_features, gamma, selected_lambda, Gs, logger)

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
