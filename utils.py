import numpy as np
from pprint import pprint

def compute_A_inv_gradient(
						   A:np.ndarray, 
						   b:np.ndarray, 
						   z:np.ndarray, 
						   z_grad:np.ndarray, 
						   Phi:np.ndarray
						   ) -> np.ndarray:

	A_inv = np.linalg.pinv(A)
	##Inner sum:
	sum_inner = 0
	for i in range(Phi.shape[0]-1):
		sum_inner += z[i] @ (Phi[i]-Phi[i+1])

	ret = -1 * A_inv @ (1.0 /(Phi.shape[0]-1) * sum_inner) @ A_inv

	return ret

def compute_b_grandient(z_grad:np.ndarray, 
						rewards:np.array
						) -> np.array:
	ret = 0
	sum_inner = 0
	for i in range(z_grad.shape[0]):
		sum_inner += z[i] * rewards[i]

	return sum_inner * 1.0 / (z_grad.shape[0])

def compute_z_gradient()
def compute_hjj_gradient()
def compute_epsilon_lambda_gradient()
def compute_lcv_h_gradient()
