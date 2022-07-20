#!/usr/bin/env python
# coding: utf-8

import struct
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

path = os.path.join(os.path.expanduser('.'), 'MNIST')

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

imagefile = './MNIST/train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

plt.imshow(imagearray[0], cmap=plt.cm.binary)

plt.show()

class Perceptron:
    def __init__(self, X_path, Y_path, eta = 1, activation_funtion = None, dActivation = None, epoch = 500):
        
        if (activation_funtion == None):
            self.activation_funtion = self.sigmoid
            self.dActivation = self.dSigmoid
        else:
            self.activation_funtion = activation_funtion
            self.dActivation = dActivation
            
        self.epoch = 5000
    
        self.X_train = self.imageProcess(self.read_idx(path+'/train-images.idx3-ubyte'))
        y = self.read_idx(path+'/train-labels-idx1-ubyte')
        self.y_train = self.oneHotEncoding(self.read_idx(path+'/train-labels-idx1-ubyte'))
        
        self.n_x = self.X_train.shape[0]
        print("n_x  = {}".format(self.n_x))
        self.n_h = 100
        
        np.random.seed(5)
        
        self.eta = {'W1' : np.zeros((self.n_h, self.n_x)) + 0.05,
                    'b1' : np.zeros((self.n_h, 1)) + 0.05,
                    'W2' : np.zeros((10, self.n_h)) + 0.05,
                    'b2' : np.zeros((10, 1)) + 0.05,
                 }
        self.NParams = {'W1': np.random.randn(self.n_h, self.n_x)* np.sqrt(1. / self.n_x),
                 'b1': np.zeros((self.n_h, 1)),
                 'W2': np.random.randn(10, self.n_h)* np.sqrt(1. / self.n_h),
                 'b2': np.zeros((10, 1))
                 }
    
    def read_idx(self, filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        
    def oneHotEncoding(self, label):
        n = np.max(label)+1
        v = np.eye(n)[label]
        print('n  {}'.format(n))
        print(v)
        return v.T
        
    def lost(self, y, y_hat):#, lamda, params):
        m = y.shape[1]
        cost = -(1/m) * np.sum(y*np.log(y_hat)) #+ lamda/(2*m) * (np.sum(params['W1']**2) + np.sum(params['W2']**2))
        return cost


    def imageProcess(self, data):
        data = data/255
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
        return data.T

    def softMax(self, X):
        e = np.exp(X)
        p = e/np.sum(e, axis=0)
        return p

    def tanh(self,z):
        return np.tanh(z)

    def dTanh(self, z):
        return 1/(np.cosh(z)**2)
        
    def sigmoid(self, z):
        return 1./(1.+np.exp(-z))
       
    def dSigmoid(self, z, alpha = 1):
        return alpha * sigmoid(z) *(1-sigmoid (z))

    def forward(self, X, params, activation):
        forwardRes = {}
        #print("*"*100)
        #print(params['W1'].shape)
        ##print(X.shape)
        #print(params['b1'].shape)
        forwardRes['Z1'] = np.matmul(params['W1'], X) + params['b1']
        forwardRes['A1'] = activation(forwardRes['Z1'])
        forwardRes['Z2'] = np.matmul(params['W2'],forwardRes['A1']) + params['b2']
        forwardRes['A2'] = self.softMax(forwardRes['Z2'])
        return forwardRes


    def back(self, X, y,forwardRes, params,dActivation):
        m = X.shape[1]
        gradient = {}
        #e2
        gradient['dZ2'] = forwardRes['A2'] - y
        #E
        gradient['E'] = np.sum(gradient['dZ2'] ** 2) / (2 * m)
        #dJ/dW2
        gradient['dW2'] = (1./m) * np.matmul(gradient['dZ2'], forwardRes['A1'].T)
        #dJ/db2
        gradient['db2'] = (1./m) * np.sum(gradient['dZ2'], axis=1, keepdims=True)
        
        gradient['dA1'] = np.matmul(params['W2'].T, gradient['dZ2'])
        #e
        gradient['dZ1'] = gradient['dA1'] * dActivation(forwardRes['Z1'])
        #dJ/dW1
        gradient['dW1'] = (1./m) * np.matmul(gradient['dZ1'], X.T)
        #dJ/db1
        gradient['db1'] = (1./m) * np.sum(gradient['dZ1'])
        return gradient

    def updater(self, params,grad,eta):
        updatedParams = {}
        #print("Befor Update", params['b1'].shape) 
        #print(grad['db1'].shape)
        #print(eta.shape)
        updatedParams['W2'] = params['W2'] - grad['dW2'] * eta['W2']
        updatedParams['b2'] = params['b2'] - grad['db2'] * eta['b2']
        updatedParams['W1'] = params['W1'] - grad['dW1'] * eta['W1']
        updatedParams['b1'] = params['b1'] - grad['db1'] * eta['b1']
        
        #print("++\t\tUpdate: ", updatedParams['b1'].shape)
        return updatedParams

    def classifer(self, X, params,activation):
        Z1 = np.matmul(params['W1'], X) + params['b1']
        A1 = activation(Z1)
        Z2 = np.matmul(params['W2'],A1) + params['b2']
        A2 = self.softMax(Z2)
        pred = np.argmax(A2, axis=0)
        return pred
        
    def wrong_classifications(self, Y, Y_hat):
        count = 0
        R1 = np.argmax(Y_hat, axis=0)
        for y, index in zip(Y, R1):
                if y[index] != 1:
                    count += 1
                    #print( y[i], y_hat[i])
        return count
        
    def mymult(self, matrix, array):
        res = [a * i for a, i in zip(matrix, array)]
        return np.array(res)
        
    def train(self, m = 10000):
        
        epoch = self.epoch
        
        i = -1
        k = 0
        stability_time = 0 
        min_wrong_classifications = 60000 * 10
        min_loss = None
        eta_before = None
        gradient_before = None
        isStop = False
        
        Ebefore = None
        delta_eta = {'W1' : None,
                     'b1' : None,
                     'W2' : None,
                     'b2' : None,
                    }
              
        test_num = int(m * 0.1)
        idx_test = np.random.permutation(self.X_train.shape[1])[:test_num]
        X_test = self.X_train[:,idx_test]
        Y_test = self.y_train[:,idx_test]
        
        while (i < epoch) and not isStop :
            i += 1
        #for i in range(epoch):
            idx = np.random.permutation(self.X_train.shape[1])[:m]
            X=self.X_train[:,idx] 
            y=self.y_train[:,idx]
                
            
            
            
            
            forwardRes = self.forward(X, self.NParams,self.activation_funtion)
            gradient = self.back(X, y, forwardRes, self.NParams, self.dActivation)
            
            
            forwardTest = self.forward(X_test, self.NParams,self.activation_funtion)
            gradientTest = self.back(X_test, Y_test, forwardTest, self.NParams, self.dActivation)
            
            E = gradientTest['E']
            if (Ebefore and E > Ebefore):
                stability_time += 1
                print("*" * 100)
                print("ENd after {} epochs".format(i))
                print("E = {:10.7f}, E_before = {:10.7f} Delta = {:10.7f}".format(E, Ebefore, E - Ebefore))

                
            else:
                stability_time = 0
            
            
            if stability_time > 3:
                isStop = True
                
            coef = 1
            if (k > 0):
                #print("self.eta", self.eta.shape)
                delta_eta['W1'] = coef * (gradient['dW1'] / self.eta['W1']) * (gradient_before['dW1'] / eta_before['W1'])
                delta_eta['b1'] = coef * (gradient['db1'] / self.eta['b1']) * (gradient_before['db1'] / eta_before['b1'])
                delta_eta['W2'] = coef * (gradient['dW2'] / self.eta['W2']) * (gradient_before['dW2'] / eta_before['W2'])
                delta_eta['b2'] = coef * (gradient['db2'] / self.eta['b2']) * (gradient_before['db2'] / eta_before['b2'])
                
                
                
                
                self.eta['W1'] = self.eta['W1']  - delta_eta['W1']
                self.eta['b1'] = self.eta['b1']  - delta_eta['b1']
                self.eta['W2'] = self.eta['W2']  - delta_eta['W2']
                self.eta['b2'] = self.eta['b2']  - delta_eta['b2']
                #print("DELTA_ETA = ", delta_eta.shape)
                #print('gradient[dW1]', gradient['dW1'].shape)
                
                
            
            wrong_classifications = self.wrong_classifications(y,forwardRes['A2'])
            if (wrong_classifications < min_wrong_classifications):
                min_wrong_classifications = wrong_classifications
                stability_time = 0
            else:
                stability_time += 1
            
            loss = self.lost(y, forwardRes['A2'])
            
            if i % 10 == 0:
                print("\n\nTraining {} epochs %  Lost = {:7.2f}".format(i, loss))
                if (Ebefore):
                    print("E = {:10.7f}, E_before = {:10.7f} Delta = {:10.7f}".format(E, Ebefore, E - Ebefore))
            
            self.NParams = self.updater(self.NParams, gradient, self.eta)
            gradient_before = gradient
            eta_before = self.eta
            #print("E = {}, E_before = {}".format(E, Ebefore))
            Ebefore = E
            
            
        
        print("\n\nFinish Traning")
             
    def test(self, fileData = None, fileLabel = None):
        if (fileData == None or fileLabel == None):
            return
        X_test = self.imageProcess(self.read_idx(fileData))
        y_test = self.read_idx(fileLabel)
        y_hat = self.classifer(X_test, self.NParams, self.activation_funtion)
        isErr = False
        count = 0
        while (not isErr):
            isErr = not (y_hat[count] == y_test[count])
            count += 1
        print("Error from Test [{}] / [{}]".format(count, count / len(y_test) * 100))
        print('Accuracy:',sum(y_hat==y_test)*1/len(y_test) * 100)
   

def tanh(z):
        return np.tanh(z)

def dTanh(z):
    return 1/(np.cosh(z)**2)  

def sigmoid(z):
        return 1./(1.+np.exp(-z))
       
def dSigmoid(self, z):
    return sigmoid(z) *(1-sigmoid (z))    

network = Perceptron(X_path = path+'/train-images.idx3-ubyte', Y_path = path+'/train-labels-idx1-ubyte', eta = 0.1, activation_funtion = tanh, dActivation = dTanh)
network.train()
network.test(path+'/t10k-images-idx3-ubyte', path+'/t10k-labels-idx1-ubyte')
    
    
