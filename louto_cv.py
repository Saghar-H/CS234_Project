import numpy as np

class LOuTOCV:
  """
  Leave one tuple out cross validation
  """
  def __init__(self, lstd_lambda, theta):
    """
    Initialize parameters lambda and theta
    """
    self.lstd_lambda = lstd_lambda
    self.theta = theta

  def compute_cv_gradient(self):
  def incremental_update(self, epsilon):
    """
    while not converge:
    sample(s0, a0, r0, s1, a1, r1, ..., sT-1, aT-1, rT-1, sT)
    calculate gradient CV withe respect to lambda
    update lambda
    update theta
    """
	old_theta = self.theta
	while(self.theta - old_theta):
        
    
if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    theta, lstd_lambda = incremental_update

