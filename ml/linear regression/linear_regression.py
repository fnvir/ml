def f(x:list):
    z=[]
    for i in x:
        z.append([1]+i if type(i)==list else [1,i])
    return z


class linearRegression:
    def __init__(self,x,y,alpha=0.01):
        self.x=f(x)
        self.y=y
        self.m=len(self.x)
        self.alpha=alpha
        self.theta=[0]*len(self.x[0])

    # cost func
    def J(self):
        return sum((self.h(self.x[i])-self.y[i])**2 for i in range(self.m))/(2*self.m)

    # hypothesis func
    def h(self,x:list):
        return sum(x[i]*self.theta[i] for i in range(len(x)))

    def gradient_descent(self,max_iter=5000):
        a_m=self.alpha/self.m
        for _ in range(max_iter):
            for j in range(len(self.theta)):
                for i in range(len(self.x)):
                    self.theta[j]-=a_m*(self.h(self.x[i])-self.y[i])*self.x[i][j]
            # if _%1000==0: print(f'Iter {_}, Cost: {self.J()}')

    def show(self):
        print(self.theta)
        print(self.J())






x=[[21],[16],[24],[14],[30]]
y=[40,30,36,22,54]
r=linearRegression(x,y,0.01)
r.gradient_descent(5000)
r.show()