import math
import pdb

class ADAM:
    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-4):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features
        epsilon : float
            To avoid having the `A` matrix be singular, it is sometimes helpful
            to initialize it with the identity matrix multiplied by `epsilon`.
        """
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.x = 0						#initialize the vector
        self.m_t = 0
        self.v_t = 0

    def update(self, grad, t):
        g_t = grad		#computes the gradient of the stochastic function
        self.m_t = self.beta_1*self.m_t + (1-self.beta_1)*g_t	#updates the moving averages of the gradient
        self.v_t = self.beta_2*self.v_t + (1-self.beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
        m_cap = self.m_t/(1-(self.beta_1**t))		#calculates the bias-corrected estimates
        v_cap = self.v_t/(1-(self.beta_2**t))		#calculates the bias-corrected estimates
        self.x = self.x - (self.alpha*m_cap)/(math.sqrt(v_cap) + self.epsilon)	#updates the parameters
