import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.vw = np.zeros((output_size, input_size)) # weight velocity for momentum
        self.vb = np.zeros((output_size, 1)) # bias velocity
        self.inp=None
        self.Z=None
        self.activation=activation

    def predict(self,X):
        return self.f(np.dot(self.weights,X)+self.bias)

    def f(self,z):
        if self.activation is None: return z
        return self.activation.call(z)

    def forward(self,inp):
        self.inp=inp
        self.Z=np.dot(self.weights,inp)+self.bias
        return self.f(self.Z)

    def backward(self, output_gradient, alpha, beta, gamma): # alpha: learning rate, beta: momentum, gamma: regularization
        if self.activation is not None:
            output_gradient*=self.activation.call(self.Z,True) # activation func backprop
        dW=np.dot(output_gradient, self.inp.T) # weight gradient: dE/dW (E=cost func)
        dB=output_gradient # dE/dB = dE/dY
        input_gradient=np.dot(self.weights.T,output_gradient) # dE/dX 
        self.vw=beta*self.vw+(1-beta)*dW
        self.vb=beta*self.vb+(1-beta)*dB
        self.weights*=(1-(alpha*gamma/np.size(output_gradient))) #L2 regularization
        self.weights-=alpha*self.vw
        self.bias-=alpha*self.vb
        # if not beta: #no momentum
        # self.weights-=alpha*dW
        # self.bias-=alpha*dB
        return input_gradient
