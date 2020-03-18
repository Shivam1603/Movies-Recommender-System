# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:05:15 2018

@author: Shivam
"""
import numpy as np
import pandas as pd
def cofiCostFun(params, Y, R, num_users, num_movies, num_features, regParam):
    J = 0.0

    X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))
     
    #without reg
    temp = np.zeros(Y.shape)
    temp = np.dot(X,Theta.T)
    temp1 = temp*R - Y*R
    temp2 = temp1*temp1
    temp3 = np.sum(np.sum(temp2, axis = 0), axis = 0)
    J_init = 0.5 * temp3
    #with reg
    reg_theta = 0
    reg_theta = np.sum(np.sum(Theta*Theta, axis=0), axis=0)
    reg_theta = 0.5*regParam*reg_theta
    
    reg_X = 0
    reg_X = np.sum(np.sum(X*X, axis=0), axis=0)
    reg_X = 0.5*regParam*reg_X
    
    J = J_init + reg_theta + reg_X
    return J

def cofiGrad(params, Y, R, num_users, num_movies, num_features, regParam):
   
    X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)    
    #without reg
    temp = np.zeros(Y.shape)
    temp = np.dot(X,Theta.T)
    temp1 = temp*R - Y*R
    
    X_grad = np.dot(temp1,Theta) + regParam*X
    Theta_grad = np.dot(temp1.T,X) + regParam*Theta   
    
    grad = np.concatenate((X_grad,Theta_grad),axis = 0)
    grad = np.reshape(grad, (grad.shape[0]*grad.shape[1],1))
    
    grad = np.ndarray.flatten(grad)
    return grad    

def checkCostFunc(regParam):
    numgrad = computeNumericalGradient(J, params_test, regParam)
    _, analytical_grad = cofiCostFun(params_test, Y.values[0:5, 0:4], R.values[0:5, 0:4], 4, 5, 3, regParam)
    print(numgrad)
    print(analytical_grad)
    return None
def computeNumericalGradient(J,theta, regParam):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(0,theta.shape[0]):
        perturb[p] = e
        loss1,_ = cofiCostFun(theta-perturb, Y.values[0:5,0:4], R.values[0:5,0:4], 4, 5, 3, regParam)
        loss2,_ = cofiCostFun(theta+perturb, Y.values[0:5,0:4], R.values[0:5,0:4], 4, 5, 3, regParam)
        numgrad[p] = (loss2 - loss1)/(2*e)
        perturb[p] = 0
    return numgrad

#checkCostFunc(1.5)
def loadMovieList():
    file = open("movie_ids.txt", "r")
    n = 1682
    movieList = []
    movieName = []
    
    line = file.readlines()
    for i in range(0,n):
        movieName.append(line[i].split(" ", 1))
        movieList.append(movieName[i][1])
    file.close()
    return movieList

def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros((Y.shape))
    for i in range(0,m):
        idx = np.nonzero(R[i,:])
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm,Ymean
