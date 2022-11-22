import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class linear_regression:
    def __init__(self,X,Y):
        self.x=np.hstack((np.ones((len(X),1)),np.array(X)))
        self.y=np.array(Y).reshape(-1,1)
        self.theta=np.zeros((self.x.shape[1],1))
        self.m=self.x.shape[0]

    # cost func
    def J(self):
        return np.sum((self.x.dot(self.theta)-self.y)**2)/(2*self.m)

    # hypothesis func
    def h(self,z):
        z=np.hstack((np.ones((len(z),1)), np.array(z)))
        return z.dot(self.theta)

    def gradient_descent(self,max_iter=5000,alpha=0.001):
        a_m=alpha/self.m
        for _ in range(max_iter):
            self.theta -= a_m*np.sum(np.multiply(self.x.dot(self.theta)-self.y, self.x),axis=0).reshape(-1,1)

    def show(self):
        print(self.theta)
        print(self.J())





df = pd.read_csv("data.csv", header=None)
df.rename(columns={0: 'population', 1: 'profit'}, inplace=True)

x = df['population'].values.reshape(-1, 1)
y = df['profit'].values.reshape(-1, 1)
lr=linear_regression(x,y)
lr.gradient_descent(5000,0.01)
lr.show()

