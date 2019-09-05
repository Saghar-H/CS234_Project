import numpy as np
import pdb
from pprint import pprint

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
                           _lambda, 
                            gamma,
						   A:np.ndarray, 
						   Phi:np.ndarray,
						   ep_states: np.ndarray,
						   ) -> np.ndarray:
    '''
    inputs: 
    A: dxd
    Z_grad: dxT
    Phi: sxd
    ep_states: Tx1

    return: 
    gradient of A inverse: dxd
    '''

    A_inv = np.linalg.pinv(A)
    ##Inner sum:
    sum_inner = 0
    for i in range(len(ep_states)-1):
        z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, i)
        sum_inner += z_grad @ (Phi[ep_states[i],:]-Phi[ep_states[i+1],:])

    ret = -1 * A_inv @ (1.0 /(Phi.shape[0]-1) * sum_inner) @ A_inv

    return ret

def compute_b_gradient( 
                        _lambda, 
                        gamma, 
                        Phi, 
                        ep_states, 
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
    z = compute_z(_lambda, Phi, ep_states, j)
    z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, j)
    A_inv_grad = compute_A_inv_gradient(A, b, z_grad, Phi)
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
    A_inv_grad = compute_A_inv_gradient(A, b, z_grad, Phi)
    b_grad = compute_b_gradient(z_grad, rewards)
    term1 = -(Phi[cur_state, :]-gamma* Phi[next_state, :])
    term2 = A_inv_grad @ b
    term3 = A_inv @ b_grad
    term4 = term1 @ (term2 + term3)
    return term4

def compute_lcv_lambda_gradient(epsilon, H, ep_states, epsilon_lambda_gradient, H_gradient):
    result = 0
    T = len(ep_states)
    for t in range(T):
        s_t = ep_states[t]
        I_H = 1 - H[s_t, s_t]
    result += (2 * epsilon[t])/(I_H) * (epsilon_lambda_gradient[t] / I_H + (2*epsilon[t]*H_gradient[s_t,s_t]) / (I_H**2))
    return result
