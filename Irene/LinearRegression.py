import numpy as np


class LinearRegression(object):
    '''
    Class for Linear Regression
    '''
    def __init__(self,X):
        self.X = X

    def train(self):
        pass

    def test(self):
        pass


def loadData(fileName):
    X = []
    file = open(fileName,'r')
    for line in file:
        if len(line) > 0:
            X.append([int(x) for x in line.split(',')])

    return np.asarray(X)

if __name__=='__main__':
    X = loadData('./regression.csv')
    lr = LinearRegression(X)
    lr.train()
    lr.test()
    print('Training Finish. ')
