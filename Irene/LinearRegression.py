# data from http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#linear-regression-example
# 
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

        # y = Wx+b
        self.X = X
        self.y = np.expand_dims(y, axis=1)
        self.W = np.ones((np.shape(self.X)[1], np.shape(self.X)[1]))
        self.b = np.ones(shape=(1,1))

        # initial prediction
        self.predict = np.matmul(self.W, np.transpose(self.X)) + self.b

        print('Started training...')

        # mean square error
        self.error = np.sum(np.square(np.subtract(self.y, np.transpose(self.predict))))/(2*len(self.X))
        # use SGD to minimize the error
        self._SGD()
        print('Finished training...')




    def _SGD(self, learning_rate1 = 10., learning_rate2 = 0.001):
        #run SGD to find optimum params


        # threshold set to be 0.001
        while (self.error > 0.001):
            # randomly sample a data record to update
            i = np.random.randint(0, len(self.X))
            temp = np.sum(np.subtract(self.y, np.transpose(self.predict)))/len(self.X)
            
            # update W
            self.WG = temp * self.X[i]
            self.W += learning_rate1 * self.WG

            # update b
            self.bG = temp
            self.b += learning_rate2 * self.bG

            # update error
            tempPredict = np.matmul(self.W, np.transpose(self.X)) + self.b
            tempError = np.sum(np.subtract(self.y, np.transpose(tempPredict))) / (2 * len(self.X))

            #print ('---', self.W,self.b,tempError)
            self.error = tempError






    def test(self, X,y):
        
        # apply the model, show the parameters and error
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
    
    # training
    lrm.train(diabetes_X_train, diabetes_y_train)

    # getting prediction
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

