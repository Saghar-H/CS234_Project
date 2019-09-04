import numpy as np
from pprint import pprint

def compute_A_inv_gradient(
						   A:np.ndarray, 
						   z:np.ndarray, 
						   z_grad:np.ndarray, 
						   Phi:np.ndarray
						   ) -> np.ndarray:
	'''
	inputs: 
	A: dxd
	z: dxT
	z_grad: dx1
	Phi: Txd

	return: 
	gradient of A inverse: dxd
	'''

	A_inv = np.linalg.pinv(A)
	##Inner sum:
	sum_inner = 0
	for i in range(Phi.shape[0]-1):
		sum_inner += z[:,i] @ (Phi[i,:]-Phi[i+1,:])

	ret = -1 * A_inv @ (1.0 /(Phi.shape[0]-1) * sum_inner) @ A_inv

	return ret

def compute_b_gradient(z_grad:np.ndarray, 
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
	for i in range(z_grad.shape[1]):
		sum_inner += z[:,i] * rewards[i]

	return sum_inner * 1.0 / (z_grad.shape[1])

def compute_z_gradient(_lambda, gamma, Phi, ep_states, j):
    result = 0
    for i in range(j):
	    result += (j-i)* (gamma ** (j-i)) * (_lambda ** (j-i-1)) * Phi[ep_states[i]]
    return result

def compute_hjj_gradient(Phi, _lambda, gamma, ep_states, j, A, b,  A_inv, z, cur_state, next_state):
    z_grad = compute_z_gradient(_lambda, gamma, Phi, ep_states, j)
    A_inv_grad = compute_A_inv_gradient(A, b, z, Phi)
    term1 = Phi[cur_state, :]-gamma* Phi[next_state, :]
    term2 = term1 @ A_inv
    term3 = term2 @ z_grad
    term4 = term1 @ A_inv_grad
    term5 = term4 @ z
    return term3 + term5


def compute_epsilon_lambda_gradient(Phi, gamma, A, b,  A_inv, z, z_grad, cur_state, next_state, rewards):
    A_inv_grad = compute_A_inv_gradient(A, b, z, z_grad, Phi)
    b_grad = compute_b_gradient(z_grad, rewards)
    term1 = -(Phi[cur_state, :]-gamma* Phi[next_state, :])
    term2 = A_inv_grad @ b
    term3 = A_inv @ b_grad
    term4 = term1 @ (term2 + term3)
    return term4

def compute_lcv_lambda_gradient(epsilon, H, T, ep_states, epsilon_lambda_gradient, H_gradient)
    result = 0
    for t in range(T)
        s_t = ep_states[t]
        1_H = 1 - H[s_t, s_t]
	    result += (2 * epsilon[t])/(1_H) * (epsilon_lambda_gradient[t] / 1_H + (2*epsilon[t]*H_gradient[s_t,s_t]) / (1_H**2))
    return result
