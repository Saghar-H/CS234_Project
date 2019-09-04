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
def compute_lcv_h_gradient()
