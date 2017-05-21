import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class LinearRegression(object):
    '''
    Class for Linear Regression
    '''
    def __init__(self):
        self.X = None
        self.y = None

    def train(self, X, y):

        # y =Wx+b
        self.X = X
        self.y = np.expand_dims(y, axis=1)
        self.W = np.ones((np.shape(self.X)[1], np.shape(self.X)[1]))
        self.b = np.ones(shape=(1,1))


        self.predict = np.matmul(self.W, np.transpose(self.X)) + self.b

        print('Started training...')

        #error: euclidean distance
        self.error = np.sum(np.square(np.subtract(self.y, np.transpose(self.predict))))/(2*len(self.X))
        # SGD
        self._SGD()
        print('Finished training...')




    def _SGD(self, learning_rate1 = 10., learning_rate2 = 0.001):
        #run SGD to find optimum params



        while (self.error > 0.001):
        # for i in range(len(self.X)):
            # print ('---',self.error)
            i = np.random.randint(0, len(self.X))

            temp = np.sum(np.subtract(self.y, np.transpose(self.predict)))/len(self.X)
            # update gradient
            self.WG = temp * self.X[i]
            self.W += learning_rate1 * self.WG

            # update b
            self.bG = temp
            self.b += learning_rate2 * self.bG

            # update error
            tempPredict = np.matmul(self.W, np.transpose(self.X)) + self.b
            tempError = np.sum(np.subtract(self.y, np.transpose(tempPredict))) / (2 * len(self.X))

            print ('---', self.W,self.b,tempError)
            self.error = tempError






    def test(self, X,y):
        # apply the model, show the error
        predict = np.matmul(self.W, np.transpose(X)) + self.b
        error = np.linalg.norm(y - np.transpose(predict))

        print('W = {}, b ={}.'.format(self.W, self.b))
        print('Testing error is {}.'.format(error))

        return np.transpose(predict)




def loadData(fileName=None):


    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    return diabetes_X_train,diabetes_X_test,diabetes_y_train,diabetes_y_test

if __name__=='__main__':




    # loading data
    diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = loadData(None)
    lrm = LinearRegression()

    lrm.train(diabetes_X_train, diabetes_y_train)

    #base:2548.07
    #base: [ 938.23786125] 152.918861826
    predicts = lrm.test(diabetes_X_test, diabetes_y_test)
    #lrm.predict(diabetes_X_test)

    print('Training Finish. ')

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, np.squeeze(predicts), color='blue',
             linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    #W = [[ 733.3663413]], b =[[ 153.05736216]].
    #Testing error is 1593.5608349927609.

