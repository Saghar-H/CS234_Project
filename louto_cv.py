import numpy as np

def compute_cv_gradient(phi, theta, gamma, lstd_lambda, P, V, D):
    I = np.eye(len(P), len(P[0]))
    phi_t = phi.transpose()
    print(I)
    print(P)
    print(gamma*lstd_lambda*P)
    print(I - gamma*lstd_lambda*P)
    inv1 = np.linalg.inv(I - gamma*lstd_lambda*P)
    psi = np.dot(np.dot(np.dot(phi_t, D), inv1), I - gamma*P) 
    print(psi)
    inv2 = np.linalg.inv(np.dot(psi, phi))
    H = np.dot(np.dot(phi, inv2), psi)
    print(H)
	gradient_psi = gamma * phi_t @ D
               
if __name__ == '__main__':
    nS = 5
    nF = 3
    phi = np.random.rand(nS, nF)
    theta = np.random.rand(nF, 1)
    lstd_lambda = 0.5
    gamma = 1
    P = np.random.rand(nS, nS)
    V = np.random.rand(nS, 1)
    D = np.random.rand(nS, nS)
    compute_cv_gradient(phi, theta, gamma, lstd_lambda, P, V, D)
