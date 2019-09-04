import numpy as np
from pprint import pprint

def compute_A_inv_gradient(A, b, z, Phi)
def compute_b_grandient(b)
def compute_z_gradient(_lambda, gamma, Phi, ep_states, j):
    result = 0
    for i in range(j):
	    result += (j-i)* (gamma ** (j-i)) * (_lambda ** (j-i-1)) * Phi[ep_states[i]]
    return result
def compute_hjj_gradient()
def compute_epsilon_lambda_gradient()
def compute_lcv_lambda_gradient(epsilon, H, T, ep_states, epsilon_lambda_gradient, H_gradient)
    result = 0
    for t in range(T)
        s_t = ep_states[t]
        1_H = 1 - H[s_t, s_t]
	    result += (2 * epsilon[t])/(1_H) * (epsilon_lambda_gradient[t] / 1_H + (2*epsilon[t]*H_gradient[s_t,s_t]) / (1_H**2))
    return result