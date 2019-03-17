import gym
import gym_walk
import pdb
import numpy as np
import random
import copy
from lstd import LSTD
from pprint import pprint


def get_discounted_return(episode_rewards, gamma):
    discounted_rewards = [0] * (len(episode_rewards) + 1)
    for i in range(len(episode_rewards) - 1, -1, -1):
        discounted_rewards[i] = discounted_rewards[i + 1] * gamma + episode_rewards[i]
    return discounted_rewards[:-1]


def init_env(env_name, seed):
    env = gym.make(env_name)
    env.reset()
    random.seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    return env


def compute_P(transition_probs, num_actions, num_states):
    ret = np.zeros((num_states, num_states))
    for s in transition_probs.keys():
        for a in transition_probs[s]:
            for tup in transition_probs[s][a]:
                sp = tup[1]
                p_sasp = tup[0]
                ret[s, sp] += 1.0 / num_actions * p_sasp

    return ret


def compute_cv_gradient(phi, theta, gamma, lstd_lambda, P, V, D):
    I = np.eye(len(P), len(P[0]))
    phi_t = phi.transpose()
    V_t = V.transpose()
    I_gamma_P = I - gamma * P
    print('*****---- P ----*****')
    print(P)
    inv1 = np.linalg.inv(I - gamma * lstd_lambda * P)
    psi = phi_t @ D @ inv1 @ I_gamma_P
    print('*****---- psi ----*****')
    print(psi)
    inv2 = np.linalg.inv(psi @ phi)
    H = phi @ inv2 @ psi
    I_H = I - H
    d = np.diag(np.diag(I_H))
    print('*****----H----*****')
    print(H)
    print('*****----diag(I-H)----*****')
    print(d)
    d_inv1 = np.linalg.inv(d)
    print('*****----(diag(I-H))^(-1)----*****')
    print(d_inv1)
    d_inv2 = np.linalg.inv(d_inv1)
    print('*****----(diag(I-H))^(-2)----*****')
    print(d_inv2)
    psi_gradient = gamma * phi_t @ D @ inv1 @ P @ inv1 @ I_gamma_P
    H_gradient = phi @ inv2 @ (psi_gradient - psi_gradient @ phi @ inv2 @ psi)
    diag_H_gradient = np.diag(np.diag(H_gradient))
    d_inv2_gradient = d_inv1 @ (diag_H_gradient @ d_inv1 + d_inv1 @ diag_H_gradient) @ d_inv1
    print('*****---- H_gradient ----*****')
    print(H_gradient)
    H_V = phi @ theta
    term1 = -V_t @ H_gradient @ d_inv2 @ I_H @ V
    term2 = V_t @ I_H @ d_inv2_gradient @ I_H @ V
    term3 = -V_t @ I_H @ d_inv2 @ H_gradient @ V
    cv_gradient = term1 + term2 + term3
    print("#### CV Gradient #####")
    print(cv_gradient)


done = False
seed = 1358
env_name = 'WalkFiveStates-v0'
env = init_env(env_name, seed)
num_features = 5
num_states = 5
num_episodes = 100

gamma = 0.4
lambda_ = 0.2

transition_probs = env.env.P

# One hot vector representations:
Phi = np.eye(num_states)

'''
Computes monte carlo estimates of D and V:
'''
print("          Transition Probabilities:                 ")
print(transition_probs)


def run_env_episodes(num_episodes):
    D = np.ones(env.observation_space.n) * 1e-10
    V = np.zeros(env.observation_space.n)
    trajectories = {}
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
        for i in range(len(ep_states)):
            V[ep_states[i]] += ep_discountedrewards[i]

    print('Monte Carlo D:{0}'.format(D * 1.0 / total_steps, total_steps))
    print('Monte Carlo V:{0}'.format(V * 1.0 / D))
    return np.diag(D / total_steps), V / D, trajectories

def LSTD_algorithm(trajectories, num_features, gamma = 0.4, lambda_ = 0.2, epsilon=0.0):
    # LSTD operator:
    LSTD_lambda = LSTD(num_features, epsilon=0.0)
    G = []
    loss = []
    num_episodes = len(trajectories.keys())
    for ep in range(num_episodes):
        traj = trajectories[ep]
        ep_rewards = []
        ep_states = []
        episode_loss = 0
        cur_state = traj[0][0]
        LSTD_lambda.reset_boyan(cur_state)
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
            G.append(ep_discountedrewards)
            loss.append(ep_loss)
    # print('episode loss:{0}'.format(loss))
    #print(LSTD_lambda.A, LSTD_lambda.b)

def Adaptive_LSTD_algorithm(trajectories, num_features, Phi, P, V, D, lr= 0.001, gamma=0.4, lambda_0=0.2, epsilon=0.0):
    # LSTD operator:
    LSTD_lambda = LSTD(num_features, epsilon=0.0)
    G = []
    loss = []
    num_episodes = len(trajectories.keys())
    for ep in range(num_episodes):
        traj = trajectories[ep]
        ep_rewards = []
        ep_states = []
        episode_loss = 0
        cur_state = traj[0][0]
        LSTD_lambda.reset_boyan(cur_state)
        for timestep in range(len(traj)):
            cur_state, reward, next_state, done = traj[timestep]
            LSTD_lambda.update_boyan(Phi[cur_state, :], reward, Phi[next_state, :], gamma, lambda_, timestep)
            ep_rewards.append(reward)
            ep_states.append(cur_state)
        theta = LSTD_lambda.theta
        lambda_ = lambda_0 - lr * compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D)
        lambda_0 = lambda_
        ep_discountedrewards = get_discounted_return(ep_rewards, gamma)
        # print('ep_discounted:{0}'.format(ep_discountedrewards))
        if len(ep_discountedrewards) > 0:
            ep_loss = np.mean(
                [(np.dot(Phi[ep_states[t], :], theta) - ep_discountedrewards[t]) ** 2 for t in
                 range(len(ep_states))])
            # print('Episode {0} loss is {1}'.format(ep, ep_loss))
            # print('Episode {0} rewards are {1}'.format(ep, ep_rewards))
            G.append(ep_discountedrewards)
            loss.append(ep_loss)
    # print('episode loss:{0}'.format(loss))
    # print(LSTD_lambda.A, LSTD_lambda.b)
    print("average loss: ", sum(loss) / num_episodes)
    return LSTD_lambda, theta, loss, G

    print("average loss: ", sum(loss) / num_episodes)
    return LSTD_lambda, theta, loss, G

def compute_CV_loss(trajectories, num_features, gamma, lambda_, epsilon =0.0 ):
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
    for i in range(num_episodes):
        print("calculating LOTO Loss for {0} trajectory".format(i))
        traj = trajectories[i]
        for j in range(len(traj)):
            #leave one tuple oto_trajectoriesout
            loto_trajectories = copy.deepcopy(trajectories)
            del loto_trajectories[i][j]
            _, _, loss, _ = LSTD_algorithm(loto_trajectories, num_features, gamma, lambda_)
            loto_loss.append(loss)
    cv_loss = np.mean(loto_loss)
    return cv_loss


print('Generate Monte Carlo Estimates of D and V...')
D, V, trajectories = run_env_episodes(100000)
print('Done finding D and V!')

# D = np.diag([0.12443139 ,0.24981192 ,0.25088312, 0.25018808 ,0.12468549])
# V = np.array([0, 0.01776151, 0.071083, 0.26708894 ,1])

'''
Now compute the MRP value of P: P(s'|s)
'''
P = compute_P(transition_probs, env.action_space.n, env.observation_space.n)

# Run LSTD_lambda algorithm:
print('Running the LSTD Lambda Algorithm ...')
LSTD_lambda, theta, loss, G = LSTD_algorithm(trajectories, num_features,gamma, lambda_)

print("#########")
print("#### Compute CV Gradient #####")
# nS = 5
# nF = 3
# phi = np.random.rand(nS, nF)
# theta = np.random.rand(nF, 1)
# P = np.random.rand(nS, nS)
# V = np.random.rand(nS, 1)
# D = np.random.rand(nS, nS)
print(P)
# print(np.linalg.inv(P + np.ones(len(P), len(P)) * 1e-15))
print('---------theta------------')
print("Theta: {0}".format(theta))
#compute_cv_gradient(Phi, theta, gamma, lambda_, P, V, D)
cv_loss = compute_CV_loss(trajectories, num_features, gamma, lambda_ )
print("########## Compute CV Loss ###########")
print("CV Loss: {0}".format(cv_loss))