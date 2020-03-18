# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:35:20 2018

@author: Shivam
"""

import numpy as np
import pandas as pd
from scipy.optimize import fmin_cg
from helperfuncs import *
def matrices():
    dataset = pd.read_csv('ratings.csv',sep = '::', header = None,engine = 'python')
    X = dataset.iloc[:, 0:3].values
    y= np.zeros((3952,6040))
    R = np.zeros((3952,6040))
    m_id = X[:,1:2]
    u_id = X[:,0:1]
    ratings = X[:,2:3]

    for i in range(m_id.shape[0]):
        r = m_id[i]
        c= u_id[i]
        y[r-1,c-1] = ratings[i]
        R[r-1,c-1] = 1            
    return y,R
def loadMovieList():
    file = open("movies.txt", "r")
    n = 3883
    movieList = []
    movieName = []
    movieidx = []
    line = file.readlines()
    
    for i in range(0,n):
        movieName.append(line[i].split("::", 2))
        movieidx.append(movieName[i][0])
        movieList.append(movieName[i][1])
    file.close()
    return movieList,movieidx
Y,R = matrices()

u = Y.shape[1]
m = Y.shape[0]
n = 17

#my ratings:
movieList,_ = loadMovieList()   
my_ratings = np.zeros((m,1))
my_ratings_R = np.zeros((m,1))
my_ratings[315] = 5
my_ratings[1672] = 5
for i in range(0,my_ratings.shape[0]):
    if(my_ratings[i]>0):
        print("Rated " + str(my_ratings[i]) + " for " + str(movieList[i]))
        
Y = np.hstack((my_ratings, Y))
#converting my_rating to a matrix of 0/1
for j in range(0,my_ratings.shape[0]):
    if(my_ratings[i]>0):
        my_ratings_R[i] = 1

R = np.hstack((my_ratings_R, R))
# Now introduce a function normalizeRatings
 
Ynorm, Ymean = normalizeRatings(Y,R)
u = Y.shape[1]
m = Y.shape[0]
n = 17
X = np.random.randn(m,n)*0.01
Theta = np.random.randn(u,n)*0.01

X_flatten = X.reshape(X.shape[0] * X.shape[1],1)
Theta_flatten = Theta.reshape(Theta.shape[0] * Theta.shape[1],1)
initial_params = np.concatenate((X_flatten,Theta_flatten), axis = 0)
regParam = 10

theta = fmin_cg(lambda J: cofiCostFun(initial_params, Ynorm, R, u,m,n,regParam), initial_params, lambda grad: cofiGrad(initial_params, Ynorm, R, u,m,n,regParam),maxiter=100)

X = np.reshape(theta[0:m*n], (m, n))
Theta = np.reshape(theta[m*n:], (u, n))
print("Recommender system learning completed")
#PART8
p = np.dot(X,Theta.T)
my_predictions = p[:,0:1] + Ymean
    
print("Top recommendations for you:" )
my_pred_flatten = []
sort = []
idx = np.zeros((m,1))
for sublist in my_predictions:
    for item in sublist:
        my_pred_flatten.append(item)
sort = sorted(my_pred_flatten, reverse=True)
for i in range(0,m):
    idx[i] = my_pred_flatten.index(sort[i])
idx_flatten = []
for sublist in idx:
    for item in sublist:
        idx_flatten.append(item)
for i in range(0,10):    
    k = int(idx_flatten[i])
    print("predicting rating " + str(my_predictions[k]) + " for movie " + str(movieList[k]))

