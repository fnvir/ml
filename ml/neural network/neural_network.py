from layer import Layer
import activations as ac
from itertools import islice


class NeuralNetwork:
    def __init__(self,shape:tuple,cost_func,activation=ac.tanh):
        self.cost_func=cost_func
        self.network=[]
        for i in range(1,len(shape)):
            self.network.append(Layer(shape[i-1],shape[i],activation if i<len(shape)-1 else ac.tanh))

    def _forward_prop(self,inputs):
        X=inputs
        for layer in self.network:
            X=layer.forward(X)
        return X

    def _backprop(self,output_grad,alpha,beta,gamma):
        for i in range(len(self.network)-1,-1,-1):
            output_grad=self.network[i].backward(output_grad,alpha,beta,gamma)

    def gradient_descent_stochastic(self,x_train,y_train,batch_size=None,learning_rate=0.1,momentum=0.9,regularization=0.1,max_iter=1000,epsilon=1e-9,show=False):
        m=len(x_train)
        b=random_stream(m) if batch_size else None
        prev=100
        for e in range(max_iter):
            cost = 0
            z=islice(b,batch_size) if batch_size else range(m)
            j=0
            for i in z:
                # forwardprop
                x,y=x_train[i],y_train[i]
                output=self._forward_prop(x)
                # backprop
                output_grad=self.cost_func.cost_prime(y,output)
                self._backprop(output_grad,learning_rate,momentum,regularization)
                # cost
                cost += self.cost_func.cost(y, output)
                j+=1
            cost /= j
            if show:
                print(f"{e+1}/{max_iter}, cost: {cost}")
            if abs(prev-cost)<epsilon: break
            prev=cost

    def predict(self,inp):
        X=inp
        for layer in self.network:
            X=layer.predict(X)
        return X


def random_stream(m):
    '''infinite stream of distinct random ints in range [0,m)'''
    from random import shuffle
    a=[*range(m)]
    shuffle(a)
    i=0
    while 1:
        if i>=m:
            shuffle(a)
            i=0
        yield a[i]
        i+=1
