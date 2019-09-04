import numpy as np
from pprint import pprint

def compute_A_inv_gradient(A, b, z, Phi)
def compute_b_grandient(b)

def compute_hjj_gradient(Phi, gamma, A, b,  A_inv, z, cur_state, next_state):
    z_grad = compute_z_gradient(gamma, lambda_, Phi)
    A_inv_grad = compute_A_inv_gradient(A, b, z, Phi)
    term1 = Phi[cur_state, :]-gamma* Phi[next_state, :]
    term2 = term1 @ A_inv
    term3 = term2 @ z_grad
    term4 = term1 @ A_inv_grad
    term5 = term4 @ z
    return term3 + term5

def compute_z_gradient(_lambda, gamma, Phi, ep_states, j):
    result = 0
    for i in range(j):
	    result += (j-i)* (gamma ** (j-i)) * (_lambda ** (j-i-1)) * Phi[ep_states[i]]
    return result

def compute_epsilon_lambda_gradient()
def compute_lcv_h_gradient()
