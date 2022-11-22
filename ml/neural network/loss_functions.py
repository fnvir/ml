import numpy as np


class MSE:
    """Mean Squared Error"""
    @staticmethod
    def cost(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def cost_prime(y_true, y_pred):
        return (2/np.size(y_true)) * (y_pred - y_true)


class CrossEntropy:
    """Binary Cross Entropy"""
    @staticmethod
    def cost(y,_y):
        # return np.mean(-y * np.log(_y) - (1 - y) * np.log(1 - _y))
        # return -np.where(y>0,np.log(_y),np.log(1-_y)).mean()
        # return -(np.sum(np.log(_y[y==1]))+np.sum(np.log(1-_y[y==0])))/np.size(y)
        return -np.sum(np.nan_to_num(np.log(1-y+(2*y-1)*_y))/np.size(y))

    @staticmethod
    def cost_prime(y,_y):
        # return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
        # return np.where(y>0,-1/_y,1/(1-_y)) / np.size(y)
        return 1/((1-y-_y)*np.size(y))
