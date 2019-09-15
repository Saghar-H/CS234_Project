"""
Least-squares temporal difference learning, also known as LSTD(λ).
"""
import numpy as np
import scipy
import pdb

class LSTD:
    """Least-squares temporal difference learning.

  Attributes
  ----------
  n : int
      The number of features (and therefore the length of the weight vector).
  z : Vector[float]
      The eligibility trace vector.
  A : Matrix[float]
      A matrix with shape `(n, n)` that acts like a potential matrix.
  b : Vector[float]
      A vector of length `n` that accumulates the trace multiplied by the
      reward over a trajectory.
  """

    def __init__(self, n):
        """Initialize the learning algorithm.

    Parameters
    -----------
    n : int
        The number of features
    """
        self.n = n
        self.A = np.zeros((self.n, self.n))
        self.b = np.zeros(self.n)
        self.reset()

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.z = np.zeros(self.n)


    def reset_boyan(self, phi):
        """Reset weights, traces, and other parameters."""
        #self.z = np.zeros(self.n)
        self.z = phi

    @property
    def theta(self):
        """Compute the weight vector via `A^{-1} b`."""
        _theta = np.dot(np.linalg.pinv(self.A, rcond=.1), self.b)
        return _theta

    def update(self, phi, reward, phi_next, gamma, lambda_, timestep):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.

    Parameters
    ----------
    phi : array_like
        The observation/features from the current timestep.
    reward : float
        The reward from the transition.
    phi_next : array_like
        The observation/features from the next timestep.
    gamma : float
        Gamma is the discount factor for the current state.
    lambda_ : float
        Lambda is the bootstrapping parameter for the
        current timestep.
    """
        beta = 1/(1+timestep)
        self.z = (gamma * lambda_ * self.z + phi)
        self.A = (1-beta) * self.A + beta * np.inner((phi - gamma * phi_next), self.z)
        self.b = (1-beta) * self.b + beta * self.z * reward

        
    def update_boyan(self, phi, reward, phi_next, gamma, lambda_, timestep):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.

    Parameters
    ----------
    phi : array_like
        The observation/features from the current timestep.
    reward : float
        The reward from the transition.
    phi_next : array_like
        The observation/features from the next timestep.
    gamma : float
        Gamma is the discount factor for the current state.
    lambda_ : float
        Lambda is the bootstrapping parameter for the
        current timestep.
    """
        #pdb.set_trace()
        #self.A =  self.A + np.inner(np.reshape((phi - gamma * phi_next),(self.z.shape[0],1)),
        #                            np.transpose(np.reshape(self.z,(1,self.z.shape[0]))))
        self.A = self.A + + self.z.reshape((4,1))@(phi-gamma * phi_next).reshape((1,4))

        self.b = self.b +  self.z * reward
        self.z = (lambda_ * gamma * self.z + phi_next)

class MiniBatchLSTDLambda:
    """ Mini batched LSTD Lambda constructor.
    
        gamma: the discount factor
        
        lamb: the eligibility trace decay (lambda).
        
        phi: a state action projector
        
        theta_update_interval: (optional) the rate at which theta (the value
                                function parameters) are updated. By default,
                                they will be updated after every transition. 
                                This might be expensive. 
    """
    def __init__(self,
                 gamma,
                 lamb, 
                 phi,
                 theta_update_interval = None,
                 rcond = 1e-14):
        
        self.phi = phi
        
        self.b = np.zeros(self.phi.size)
        self.theta = np.zeros_like(self.b)
        self.A = np.zeros((self.phi.size, self.phi.size))
        
        self.gamma = gamma
        self.lamb = lamb
        self.z = None #np.zeros_like(self.b)
        
        self.count = 0
        
        # if update interval is negative, never update theta
        self.update_interval = theta_update_interval
        
        # if theta update interval was not specified, update theta 
        # at every step
        if theta_update_interval is None:
            self.update_interval = 1
            
        
    
        self.initialized = False
        self.rcond = rcond
    
    def __call__(self, state, action):
        phi_sa = self.phi[state, :]
        return phi_sa.dot(self.theta)
    
    def update(self, s_t, r_t, s_tp1):
        # check is episode start
        if s_t is None:
            self.z = None
            return
        
        # check if state is terminal
        if s_tp1 is None:
            phi_t = self.phi[s_t,:]
            d = phi_t
        else:
            # get feature vector (batched for better performance)
            states = np.vstack((s_t, s_tp1))
            #actions = np.vstack((a_t, a_tp1))
            
            phi_t, phi_tp1 = self.phi[s_t,:], self.phi[s_tp1,:]
            d = phi_t - self.gamma * phi_tp1
        
        if d.ndim == 1:
            d = d.reshape((1,-1))
        
        if d.shape[0] > 1:
            d = d.T
        
        
        # update traces
        if self.z is not None:
            z = self.gamma*self.lamb*self.z + phi_t
            if scipy.sparse.issparse(z):
                z.data *= (np.abs(z.data)> 1e-5)
                z.eliminate_zeros()
        else:
            z = phi_t
            
        if z.ndim == 1:
            z = z.reshape((1,-1))
            
        # update b vector
        if self.initialized:
            b = z * r_t + self.b
        else:
            b = z * r_t

        # update A matrix
        # this is done this way so that A becomes a sparse matrix if present
        
        if self.initialized:
            A = z.T.dot(d) + self.A
        else:
            A = z.T.dot(d)
            self.initialized = True
        
        # if required, update theta (without forcing an SVD)
        if self.update_interval > 0:
            self.count = (self.count + 1) % self.update_interval
            if self.count == 0:
                self.update_theta(A, b)
            
        # update saved vectors and matrices
        self.b = b
        self.z = z
        self.A = A
        
    def update_theta(self, A, b):
        try:
            if scipy.sparse.issparse(A):
                self.theta = scipy.sparse.linalg.lsmr(A,  
                                                      b.toarray().squeeze())[0]
            else:
                self.theta = np.linalg.lstsq(A, b.squeeze(), rcond = self.rcond)[0]
        except np.linalg.LinAlgError  as e:
            print ('Least-squares failed...'+ e)

        #print('z:{0}, b:{1}'.format(self.z,self.b))
