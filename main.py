import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from copy import deepcopy

class Neural_Network():
    def __init__(self):        
        # Define Hyperparameters
        self.inputLayerSize = 5
        self.num_labels = 4
        self.hiddenLayerSize = 20
        self.learning_rate = 0.01

        # randomly initialize a parameter array of the size of the full network's parameters
        self.params = (np.random.random(size=self.hiddenLayerSize * (self.inputLayerSize + 1) + self.num_labels * (self.hiddenLayerSize + 1)) - 0.5) * 0.25

    def get_params(self):
        return self.params, self.inputLayerSize, self.hiddenLayerSize, self.num_labels, self.learning_rate

    def featureNormalize(self, X):
        X_norm = X
        n = X.shape[1]
        mu = np.zeros(n, dtype=np.int)
        sigma = [0] * n

        for i in range(n):
            meanOfCurrentFeatureInX = np.mean(X_norm[:, i])
            mu[i] = meanOfCurrentFeatureInX

            X_norm[:, i] = [x - meanOfCurrentFeatureInX for x in X_norm[:, i]]

            standardDeviationOfCurrentFeatureInX = np.std(X[:, i])
            sigma[i] = standardDeviationOfCurrentFeatureInX

            X_norm[:, i] = [x / standardDeviationOfCurrentFeatureInX for x in X_norm[:, i]]

        return X_norm, mu, sigma

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagate(self, X, theta1, theta2):
        m = X.shape[0] # number of training examples

        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = a1 * theta1.T
        a2 = np.insert(self.sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = self.sigmoid(z3)

        return a1, z2, a2, z3, h

    def cost(self, params, input_size, hidden_size, num_labels, X, y, learning_rate):
        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)

        # reshape the parameter array into parameter matrices for each layer
        theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))    
        theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

        # run the feed-forward
        a1, z2, a2, z3, h = self.forward_propagate(X, theta1, theta2)

        # compute the cost
        J = 0
        for i in range(m):
            first_term = np.multiply(-y[i, :], np.log(h[i,:]))
            second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
            J += np.sum(first_term - second_term)

        J = J / m
        
        # adding regularized term
        J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

        return J

    def sigmoid_gradient(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def backprop(self, params, input_size, hidden_size, num_labels, X, y, learning_rate):  
        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)

        # reshape the parameter array into parameter matrices for each layer
        theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
        theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

        # run the feed-forward pass
        a1, z2, a2, z3, h = self.forward_propagate(X, theta1, theta2)

        # initializations
        J = 0
        delta1 = np.zeros(theta1.shape)
        delta2 = np.zeros(theta2.shape)

        # compute the cost
        for i in range(m):
            first_term = np.multiply(-y[i,:], np.log(h[i,:]))
            second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
            J += np.sum(first_term - second_term)

        J = J / m

        # add the cost regularization term
        J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

        for t in range(m):
            a1t = a1[t, :]
            z2t = z2[t, :]
            a2t = a2[t, :]
            ht = h[t, :]
            yt = y[t, :]

            d3t = ht - yt

            z2t = np.insert(z2t, 0, values=np.ones(1))
            d2t = np.multiply((theta2.T * d3t.T).T, self.sigmoid_gradient(z2t))

            delta1 = delta1 + (d2t[:, 1:]).T * a1t
            delta2 = delta2 + d3t.T * a2t

        delta1 = delta1 / m
        delta2 = delta2 / m

        # add the gradient regularization term
        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

        # unravel the gradient matrices into a single array
        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

        return J, grad

class Trainer():
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def get_data(self):
        df = pd.read_csv('data/ecommerce_data_refine.csv')
        data = df.as_matrix()
        X_all = data[:,:-1]
        y_all = data[:,-1:]

        X_train = X_all[:-300]
        y_train = y_all[:-300]

        X_test = X_all[-300:]
        y_test = y_all[-300:]

        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(y_train)

        return X_train, y_train, X_test, y_test, y_onehot


    def train(self, params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate):
        # minimize the objective function
        fmin = minimize(fun=self.N.backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 500})
        return fmin

    def predict(self, X, theta1, theta2):
        a1, z2, a2, z3, h = self.N.forward_propagate(X, theta1, theta2)  
        y_pred = np.array(np.argmax(h, axis=1) + 1) 
        return y_pred

    def process(self):
        X_train, y_train, X_test, y_test, y_onehot = self.get_data()
        X_norm, mu, sigma = self.N.featureNormalize(X_train)
        X_norm_test, mu, sigma = self.N.featureNormalize(X_test)

        m = X_train.shape[0]
        X_norm_train = np.matrix(X_norm)
        y_train = np.matrix(y_train)
        X_norm_test = np.matrix(X_norm_test)

        params, input_size, hidden_size, num_labels, learning_rate = self.N.get_params()

        fmin = self.train(params, input_size, hidden_size, num_labels, X_norm_train, y_onehot, learning_rate)

        theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
        theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

        y_pred = self.predict(X_norm_train, theta1, theta2)

        correct_train = [1 if a == b else 0 for (a, b) in zip(y_pred, y_train)]  
        accuracy_train = (sum(map(int, correct_train)) / float(len(correct_train)))  
        print('accuracy during training = {0}%'.format(accuracy_train * 100))

        y_pred_test = self.predict(X_norm_test, theta1, theta2)

        correct_test = [1 if a == b else 0 for (a, b) in zip(y_pred_test, y_test)]  
        accuracy_test = (sum(map(int, correct_test)) / float(len(correct_test)))  
        print('accuracy during testing = {0}%'.format(accuracy_test * 100))

def run():
    NN = Neural_Network()
    T = Trainer(NN)
    T.process()

if __name__ == '__main__':
    run()