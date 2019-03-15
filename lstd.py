"""
Least-squares temporal difference learning, also known as LSTD(λ).
"""
import numpy as np
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

    def __init__(self, n, epsilon=0):
        """Initialize the learning algorithm.

    Parameters
    -----------
    n : int
        The number of features
    epsilon : float
        To avoid having the `A` matrix be singular, it is sometimes helpful
        to initialize it with the identity matrix multiplied by `epsilon`.
    """
        self.n = n
        self.A = np.eye(self.n) * epsilon
        self.b = np.zeros(self.n)
        self.reset()

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.z = np.zeros(self.n)


    @property
    def theta(self):
        """Compute the weight vector via `A^{-1} b`."""
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
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

