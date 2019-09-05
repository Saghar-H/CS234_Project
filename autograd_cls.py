from jax import grad as jax_grad
from jax import jit

class AutoGrad:
    def __init__(self, loss, indx):
        self.loss = loss
        self.indx = indx # The index of paramter with respect to which we need the gradinet in the loss input parameters
    def gradient_fun(self):
        self.loss_autograd_fun = jit(jax_grad(self.loss, self.indx))
