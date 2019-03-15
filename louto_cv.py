import numpy as np

def compute_cv_gradient(phi, theta, gamma, lstd_lambda, P, V, D):
    I = np.eye(len(P), len(P[0]))
    phi_t = phi.transpose()
    V_t = V.transpose()
    inv_P = np.linalg.inv(P)
    I_gamma_P = I - gamma*P
    print(I)
    print(P)
    print(gamma*lstd_lambda*P)
    print(I - gamma*lstd_lambda*P)
    inv1 = np.linalg.inv(I - gamma*lstd_lambda*P)
    psi = phi_t @ D @ inv1 @ I_gamma_P
    print(psi)
    inv2 = np.linalg.inv(psi @ phi)
    H = phi @ inv2 @ psi
    d = np.diag(np.diag(I - H))
    d_inv2 = np.linalg.inv(np.linalg.inv(d))
    d_inv3 = np.linalg.inv(d_inv2)
    print('*****----H----*****')
    print(H)
    print('*****----diag(I-H)----*****')
    print(d)
    print('*****----(diag(I-H))^(-2)----*****')
    print(d_inv2)
    print('*****----(diag(I-H))^(-3)----*****')
    print(d_inv3)
    psi_gradient = gamma * phi_t @ D @ inv1 @ inv_P @ inv1 @ I_gamma_P
    inv3 = np.linalg.inv(psi_gradient @ phi)
    H_gradient = phi @ inv2 @ (psi_gradient - inv3 @ inv2 @ psi)
    diag_H_gradient = np.diag(np.diag(H_gradient))
    print(H_gradient)
    H_V = phi @ theta
    H_gradient_V = H_gradient @ V
    V_t_H_gradient = V_t @ H_gradient
    V_t_H = V_t @ H
    term1 = -V_t @ H_gradient @ d_inv2 @ V + V_t @ H_gradient @ d_inv2 @ H_V
    term2 = 2 * V_t @ d_inv3 @ diag_H_gradient @ V - 2 * V_t_H @ d_inv3 @ diag_H_gradient @ H_V
    term2 = term2 - 2 * V_t @ d_inv3 @ diag_H_gradient @ H_V + 2 * V_t_H @ d_inv3 @ H_V
    term3 = -V_t @ d_inv2 @ H_gradient_V + V_t_H @ d_inv2 @ H_gradient_V
    cv_gradient = term1 + term2 + term3
    print(cv_gradient)
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
