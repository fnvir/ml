import math

def add_x0(x):
    '''
    add x_0=1, for all x
    [2,3,5] -> [[1,1],[1,2],[1,3],[1,5]]
    [[2,4],[1,2] -> [[1,2,4],[1,1,2]]
    '''
    return [[1]+i if hasattr(i,'__getitem__') and type(i)!=str else [1,i] for i in x]


class LogisticRegression:
    def __init__(self,x,y,alpha=0.001):
        self.x=add_x0(x)
        self.y=y
        self.alpha=alpha
        self.m=len(self.x)
        self.theta=[0]*len(self.x[0])

    # hypothesis func
    def h(self,x):
        return self.g(sum(x[i]*self.theta[i] for i in range(len(x))))

    def predict(self,x):
        x=[1]+x # x0 = 1
        return sum(x[i]*self.theta[i] for i in range(len(x))) >= 0 # (sigmoid(z)>=0.5 only when z>=0): https://youtu.be/F_VG4LNjZZw?t=180

    def g(self,z:int):
        '''Interpreted as the probability of y=1 given that theta=x, P(y=1|x)'''
        # return math.tanh(z) # tanh
        return 1/(1+math.exp(-z)) # sigmoid

    # cost func
    def J(self):
        log=lambda z: -math.inf if z==0 else math.log(z)
        # cost=lambda x,y: -log(1-self.h(x)) if y==0 else -log(self.h(x))
        cost=lambda x,y: -log(1-y+(2*y-1)*self.h(x)) # simplified
        return sum(cost(self.x[i],self.y[i]) for i in range(self.m))# /self.m

    def gradient_descent(self,max_iter=5000):
        a_m=self.alpha/self.m
        for _ in range(max_iter):
            for j in range(len(self.theta)):
                for i in range(len(self.x)):
                    self.theta[j]-=a_m*(self.h(self.x[i])-self.y[i])*self.x[i][j]
            # print(f'iter {_}: cost {self.J()}')

#...........................................................................................


def test():
    from random import randrange
    xtrain,xtest,ytrain,ytest=[],[],[],[]
    for i in open('data2.csv'):
        a1,a2,b=map(float,i.strip().split(','))
        x,y=[a1,a2],b
        if randrange(1009)%2:
            xtrain.append(x)
            ytrain.append(y)
        else:
            xtest.append(x)
            ytest.append(y)

    lr=LogisticRegression(xtrain,ytrain,0.01464)
    print('Cost before:',lr.J())
    lr.gradient_descent(29000)
    print(f'Cost after: {lr.J()}',f'Thetas: {lr.theta}',sep='\n')

    print()
    print('train , test:',len(xtrain),len(xtest))
    N=len(xtest)
    s=sum(lr.predict(xtest[i])==ytest[i] for i in range(N))
    print(f'Accuracy: {(s/N)*100:.2f}%')


test()
