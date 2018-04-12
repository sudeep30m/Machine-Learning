# X is assumed to be a m * n np-array i.e. X = [x1,x2,x3...xm], Y is m * 1 np-array theta is n * 1 numpy array

import numpy as np
from numpy import linalg as lin
import functools
import pandas 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
import argparse

def normalize(X):
    mu = np.average(X)
    # print (mu)
    sigma = np.var(X)
    return (X - mu ) / np.sqrt(sigma)

def plotData(X , Y ):
    X = np.c_[X, Y]
    X0 = np.array(list(filter(lambda x : x[2] == 0.0, X)))
    X1 = np.array(list(filter(lambda x : x[2] == 1.0, X)))
    plt.plot(X0[:,0],X0[:,1],'x', label = "Y = Alaska")
    plt.plot(X1[:,0],X1[:,1],'_', label = "Y = Canada")
    plt.xlabel('x0')
    plt.ylabel('x1')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show(block = False)
    input("Press Enter to continue.")


def plotLine(X, Y, theta):
    plotData(X,Y)
    X2 = np.array([-2,2])
    Y2 = lineArray(theta, X2)
    plt.plot(X2, Y2, 'black')
    plt.draw()
    plt.pause(0.01)
    input("Press Enter to continue.")
    # plt.close()

def lineArray(theta, X):
    X = np.c_[np.ones(len(X)), X]
    theta0 = - theta[0] / theta[2]
    theta1 = - theta[1] / theta[2]
    theta = np.array([theta0, theta1])
    return (X.dot(theta))

def lineEquation(phi, mu0, mu1, cov):
    invCov = lin.inv(cov)
    c = np.transpose(mu1).dot(invCov).dot(mu1)
    c = c - np.transpose(mu0).dot(invCov).dot(mu0)
    theta0 = c - 2 * np.log( phi / (1 - phi ) )
    theta = 2 * ( (np.transpose(mu0) - np.transpose(mu1)).dot(invCov) )    
    return np.r_[theta0, theta]

# Plotting GDA general setting curve.  
def plotQuadraticCurve(X, Y, phi, mu0, mu1, cov0, cov1):
    # plotData(X,Y)
    X0 = np.arange(-1.5 , 2 , 0.02)
    X2 = []
    Y2 = []
    X3 = []
    Y3 = []
    for x0 in X0:
        (y1, y2) = solveGeneralCurve(x0, phi, mu0, mu1, cov0, cov1)
        if y1 is not None:
            X2.append(x0)
            Y2.append(max(y1,y2))
            X3.append(x0)
            Y3.append(min(y1,y2))
    plt.plot(X2, Y2, 'green', markersize=2)
    plt.plot(X3, Y3, 'green', markersize=2)
    plt.draw()
    plt.pause(0.01)
    input("Press Enter to continue.")
    plt.close()
    
def sigmoid(z):
    return 1 /(1 + np.exp(-z) )

# Multivariate normal distribution
def normalDistribution(x, mu, cov):
   n = len(x)
   matrix =  np.transpose(x - mu).dot(lin.inv(cov)).dot(x - mu)
   return np.exp(-matrix / 2) / ( ( (2 * np.pi) ** (n / 2) ) * ( (lin.det(cov)) ** (0.5) ) )

# Gives value of probability of y=1 given x
def hypthesisFunction(x, phi, mu0, mu1, cov0, cov1):
    p0 = normalDistribution(x, mu0, cov0)
    p1 = normalDistribution(x, mu1, cov1)
    return (p1 * phi) / ( (p1 * phi) + (p0 * (1 - phi)) )

# Solving quadratic equation
def solveQuadraticEquation(a, b, c):
    d = b ** 2 - 4 * a * c
    if d < 0: 
        return (None, None) 
    return ( (- b + np.sqrt(d)) / (2 * a) , (- b - np.sqrt(d)) / (2 * a) ) 

# Solving GDA for a general setting with different covariances. 
def solveGeneralCurve(x, phi, mu0, mu1, cov0, cov1):
    invCov0 = lin.inv(cov0) 
    invCov1 = lin.inv(cov1)

    w = 2 * np.log(phi / (1 - phi))
    w = w + np.log( lin.det(cov0) / lin.det(cov1) )
    w = w + np.transpose(mu0).dot(invCov0).dot(mu0)
    w = w - np.transpose(mu1).dot(invCov1).dot(mu1)
    w = - w

    a = invCov0[0][0]
    b = invCov0[0][1]
    c = invCov0[1][0]
    d = invCov0[1][1]
    p = invCov1[0][0]
    q = invCov1[0][1]
    r = invCov1[1][0]
    s = invCov1[1][1]
    
    u = s - d
    v = r * x + q * x - c * x -b * x
    v = v + mu0[0] * b - mu1[0] * q + mu0[0] * c - mu1[0] * r
    v = v + 2 * mu0[1] * d - 2 * mu1[1] * s 

    w = w + (p - a) * x * x
    w = w + (mu0[0] * a + mu0[1] * c - mu1[0] * p - mu1[1] * r) * x
    w = w + (mu0[0] * a + m u0[1] * b - mu1[0] * p - mu1[1] * q) * x
    return solveQuadraticEquation(u, v, w)

X = np.genfromtxt('ass1_data/q4x.dat', delimiter = '  ')
m = len(X)
Y = []
X = np.c_[ normalize(X[:,0]), normalize(X[:,1])]
with open('ass1_data/q4y.dat') as f:
    for line in f:
        word = line.split('\n')
        Y.append(word[0])
Y = list(map(lambda x : 0 if x == 'Alaska' else 1 , Y))    
Y = np.array(Y)
phi = len(Y[Y == 1]) / m
# print(phi)
X = np.c_[X, Y]
X0 = np.array(list(filter(lambda x : x[2] == 0.0, X)))
X1 = np.array(list(filter(lambda x : x[2] == 1.0, X)))
X = X[:,0:2] 
X0 = X0[:,0:2] 
X1 = X1[:,0:2] 
mu0  = np.mean(X0, axis = 0)
mu1  = np.mean(X1, axis = 0)
cov  = (np.transpose(X0 - mu0).dot(X0 - mu0) +  np.transpose(X1 - mu1).dot(X1 - mu1)) / m
cov0 = np.transpose(X0 - mu0).dot(X0 - mu0) / len(X0)
cov1 = np.transpose(X1 - mu1).dot(X1 - mu1) / len(X1)
print("mu0 = ", mu0)
print("mu1 = ", mu1)
print("cov0 = ", cov0)
print("cov1 = ", cov1)
print("cov = ", cov)
print("phi = ", phi)

# print (mu0, mu1)
# plotData(X, Y)
theta = lineEquation(phi, mu0, mu1, cov)
# print(solveGeneralCurve(80, phi, mu0, mu1, cov0, cov1))
# print(theta)
plotLine(X, Y, theta)
plotQuadraticCurve(X, Y, phi, mu0, mu1, cov0, cov1)
# print(solveQuadraticEquation(1 , 1 , 2))
# Y = np.genfromtxt('ass1_data/q4y.dat', delimiter = '')
# print(X)
# print(Y)
# m = len(X)
# initialTheta = np.zeros(3)
# epsilon = args.parameters[0]
# (theta,thetaArray) = NewtonsMethod(initialTheta, X ,Y, epsilon)
# print(theta)
# plotLine(X, Y, theta)