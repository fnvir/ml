import numpy as np


class Activation:
    @classmethod
    def call(cls,x,deriv=False):
        """call the func (deriv=call first derivative)"""
        if deriv:return cls.prime(cls,x)
        return cls.func(cls,x)

    def func(self,x)->np.ndarray:...
    def prime(self,x)->np.ndarray:...


class sigmoid(Activation):
    def func(self,x):
        return 1/(1+np.exp(-x))

    def prime(self,x):
        s=self.call(x)
        return s*(1-s)


class tanh(Activation):
    def func(self,x):
        return np.tanh(x)

    def prime(self,x):
        return 1-np.tanh(x)**2


class ReLU(Activation):
    def func(self,x):
        return x*(x>0)

    def prime(self,x):
        return (x>0)*1


class LeakyReLU(Activation):
    def func(self,x):
        return np.maximum(x*0.001, x)

    def prime(self,x):
        return np.where(x>0,1,0.001)