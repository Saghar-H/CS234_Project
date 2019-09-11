import numpy as np
import pdb
from pprint import pprint
import copy 

def compute_z(_lambda:float, 
			  gamma:float, 
			  Phi:np.ndarray, 
			  ep_states:np.ndarray, 
			  j:int
			  )-> np.ndarray:

	'''
	inputs:
	Phi: sxd
	ep_states:Tx1

	return:
	z: dx1
	'''

	ret = 0
	for i in range(j+1):
		ret += (gamma * _lambda) ** (j-i) * Phi[ep_states[i],:]

	return ret

def compute_z_gradient(_lambda, gamma, Phi, ep_states, j):
    '''
    inputs: 
    _lambda: 1x1
    gamma: 1x1
    Phi: Txd
    j: 1x1

    return: 
    gradient of z at timestep j: 1xd
    '''
    result = 0
    for i in range(j+1):
	    result += (j-i)* (gamma ** (j-i)) * (_lambda ** (j-i-1)) * Phi[ep_states[i], :]
    return result


def compute_A_inv_gradient(
                           _lambda:int, 
                            gamma: int,
						   A:np.ndarray, 
						   Phi:np.ndarray,
						   ep_states: list,
                           A_inv:np.ndarray,
						   ) -> np.ndarray:
	'''
	inputs: 
	A: dxd
	Z_grad: dxT
	Phi: sxd
	ep_states: Tx1
	A_inv: dxd

	return: 
	gradient of A inverse: dxd
	'''
	##Inner sum:
	sum_inner = 0
	for i in range(len(ep_states)-1):
	    z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, i)
	    sum_inner += z_grad.reshape((10,1)) @ (Phi[ep_states[i],:]-Phi[ep_states[i+1],:]).reshape((1,10))
	ret = -1 * A_inv @ (1.0 /(Phi.shape[0]-1) * sum_inner) @ A_inv

	return ret

def compute_b_gradient( 
                        _lambda: int, 
                        gamma: int, 
                        Phi: np.ndarray, 
                        ep_states: list, 
						rewards:np.array
						) -> np.array:

	'''
	inputs:
	z_grad: dxT
	rewards:Tx1

	return:
	gradients of b: dx1
	'''
	ret = 0
	sum_inner = 0
	for i in range(len(ep_states)):
            z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, i)
            sum_inner += z_grad * rewards[i]
	return sum_inner * 1.0 / (Phi.shape[0])

def compute_eps_t(Phi, theta, gamma, reward, ep_states, j):
    cur_state, next_state = ep_states[j], ep_states[j+1]
    term1 = Phi[cur_state, :]-gamma* Phi[next_state, :]
    eps_j = reward - term1 @ theta
    return eps_j


def compute_hjj(Phi, _lambda, gamma, ep_states, j, A_inv):
    cur_state, next_state = ep_states[j], ep_states[j+1]
    z = compute_z(_lambda,gamma,  Phi, ep_states, j)
    term1 = Phi[cur_state, :]-gamma* Phi[next_state, :]
    term2 = term1 @ A_inv
    h_jj = term2 @ z
    return h_jj


def compute_hjj_gradient(Phi, _lambda, gamma, ep_states, j, A, b,  A_inv):
    '''
    inputs:
    Phi: S X d
    _lambda: 1 X 1
    gamma : 1 X 1
    ep_states: T X 1
    j : 1 X 1
    A: d X d
    b: d X 1
    A_inv: d X d
    z: d X 1

    return:
    gradient of the H_jj wrt lambda : 1 X 1
    '''
    cur_state, next_state = ep_states[j], ep_states[j+1]
    z = compute_z(_lambda,gamma,  Phi, ep_states, j)
    z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, j)
    A_inv_grad = compute_A_inv_gradient(_lambda, gamma, A, Phi, ep_states, A_inv)
    term1 = Phi[cur_state, :]-gamma* Phi[next_state, :]
    term2 = term1 @ A_inv
    term3 = term2 @ z_grad
    term4 = term1 @ A_inv_grad
    term5 = term4 @ z
    return term3 + term5


def compute_epsilon_lambda_gradient(Phi, _lambda, gamma, A, b,  A_inv, Z, j, ep_states, rewards):
    '''
    inputs:
    Phi: S X d
    _lambda: 1 X 1
    gamma : 1 X 1
    ep_states: T X 1
    j : 1 X 1
    A: d X d
    b: d X 1
    A_inv: d X d
    Z: d X T 

    return:
    gradient of the eps_j wrt lambda : 1 X 1
    '''
    cur_state, next_state = ep_states[j], ep_states[j+1]
    z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, j)
    A_inv_grad = compute_A_inv_gradient(_lambda, gamma, A, Phi, ep_states, A_inv)
    b_grad = compute_b_gradient(_lambda, gamma, Phi, ep_states, rewards)
    term1 = -(Phi[cur_state, :]-gamma* Phi[next_state, :])
    term2 = A_inv_grad @ b
    term3 = A_inv @ b_grad
    term4 = term1 @ (term2 + term3)
    return term4

def compute_lcv_lambda_gradient(epsilon, H, ep_states, epsilon_lambda_gradient, H_gradient, grad_clip_max_norm=0):
    '''
    inputs:
    epsilon: #episodes in trajectory  x 1
    H: SxS
    ep_states: #episodes in trajectory  x 1
    epsilon_lambda_gradient: #episodes in trajectory x 1
    H_gradient: SxS
    grad_clip_max_norm: 1x1

    return:
    gradient of lcv vs lambda : float
    '''
    result = 0
    T = len(ep_states)
    for t in range(T-1):
        s_t = ep_states[t]
        I_H = 1 - H[s_t]
        result += (2 * epsilon[s_t])/(I_H) * (epsilon_lambda_gradient[s_t] / I_H + (2*epsilon[s_t]*H_gradient[s_t]) / (I_H**2))
    if grad_clip_max_norm:
        result = min(result, grad_clip_max_norm) if result > 0 else max(result, -1 * grad_clip_max_norm)
        if result in {grad_clip_max_norm, -1 * grad_clip_max_norm}:
            print('Warning! norm hit:{0}'.format(result))
    return result

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

def calculate_batch_loss(trajectories, G, theta, Phi):
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
