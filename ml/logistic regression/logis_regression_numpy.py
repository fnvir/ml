import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class logistic_regression:
    def __init__(self,X,Y:list[int]):
        self.x=self._hstack_one(np.array(X))
        self.y=np.array(Y)
        self.theta=np.zeros((self.x.shape[1]))
        self.m=self.x.shape[0]
    
    def _hstack_one(self,X):
        if len(X.shape) == 1:
            return np.hstack((np.array([1]), X))
        return np.hstack((np.ones((X.shape[0], 1)), X))

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    # cost function
    def J(self,theta=None):
        if theta is None: theta=self.theta
        z=self.x.dot(theta)
        zero=z[self.y==0]
        one=z[self.y==1]
        with np.errstate(divide='ignore'):
            return -( np.sum(np.log(self.sigmoid(one))) + np.sum(np.log(1-self.sigmoid(zero))) ) / self.m
    
    # prediction / hypothesis function
    def h(self, x):
        x=self._hstack_one(np.array(x))
        return self.sigmoid(x.dot(self.theta)) >=0.5

    def batch_gradient(self,theta=None):
        if theta is None:
            theta = self.theta
        return np.sum(np.multiply((self.sigmoid(self.x.dot(theta))-self.y).reshape((-1,1)), self.x),axis=0)

    def gradient_descent(self,alpha=.001,max_iter=5000):
        a_m=alpha/self.m
        for _ in range(max_iter):
            self.theta -= a_m*self.batch_gradient()

    def optimize(self,max_iter=5000):
        from scipy.optimize import minimize
        self.theta= minimize(fun=self.J,x0=self.theta,jac=self.batch_gradient,options={'maxiter':max_iter}).x







df = pd.read_csv('data2.csv', header=None)
df.rename(columns={0: 'exam1', 1: 'exam2', 2: 'y'}, inplace=True)
df.head()


lr = logistic_regression(df[['exam1', 'exam2']].values, df['y'].values)
print(f"Cost before converging: {lr.J():.3f}")
# optim_theta=gradient_descent(0.0016,20000)
optim_theta = lr.optimize(20000)
print(f"Cost after converging: {lr.J():.3f}")
print(lr.theta)

# # Plotting the prediction line
# col1 = "exam1"
# col2 = "exam2"
# min_ex1 = df[col1].min()
# max_ex1 = df[col1].max()
# min_ex2 = df[col2].min()
# max_ex2 = df[col2].max()
# arange_step = 0.1
# xx, yy = np.meshgrid(np.arange(min_ex1, max_ex1, arange_step),np.arange(min_ex2, max_ex2, arange_step))
# preds = np.c_[xx.ravel(), yy.ravel()]
# preds = lr.h(preds)
# preds = preds.reshape(xx.shape)
# fig = plt.figure(figsize=(12, 8))
# plt.scatter(df[df['y'] == 0][col1], df[df['y'] == 0][col2],label='Not admitted', color='yellow', edgecolor='black')
# plt.scatter(df[df['y'] == 1][col1], df[df['y'] == 1][col2],label='Admitted', marker='+', color='black')
# plt.contour(xx, yy, preds, colors='blue')
# plt.xlabel('Exam 1 score')
# plt.ylabel('Exam 2 score')
# plt.legend(loc='upper right')
# plt.title('Scores indicating admission')
# plt.show()
