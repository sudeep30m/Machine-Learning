import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def normalize(X):
    mu = np.average(X)
    # print (mu)
    sigma = np.var(X)
    return (X - mu ) / np.sqrt(sigma)

def plotData(X , Y ):
    print("\nPlotting input data. ")
    X = np.c_[X, Y]
    X0 = np.array(list(filter(lambda x : x[3] == 0.0, X)))
    X1 = np.array(list(filter(lambda x : x[3] == 1.0, X)))
    plt.plot(X0[:,1],X0[:,2],'x', label = "Y = 0")
    plt.plot(X1[:,1],X1[:,2],'o', label = "Y = 1")
    plt.xlabel('x1')
    plt.ylabel('x2')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show(block = False)
    input("Press Enter to continue.")
    # plt.close()

def plotLine(X, Y, theta):
    plotData(X,Y)
    print("\nPlotting Straight Line using theta obtained from Newton's method. ")
    X2 = np.array([-2,3])
    Y2 = lineArray(theta, X2)
    plt.plot(X2, Y2, 'b')
    plt.draw()
    plt.pause(0.01)
    input("Press Enter to continue.")
    plt.close()

def lineArray(theta, X):
    X = np.c_[np.ones(len(X)), X]
    theta0 = - theta[0]/theta[2]
    theta1 = - theta[1]/theta[2]
    theta = np.array([theta0, theta1])
    return (X.dot(theta))

#Sigmoid Function.
def sigmoid(z):
    return 1 /(1 + np.exp(-z) )

# Gradient Matrix used in Newton's method.
def gradientMatrix(theta, X ,Y):
    z = sigmoid(X.dot(theta))
    return (1/m) * np.transpose(X).dot(z - Y)

# Hessian Matrix used in Newton's method.
def hesssianMatrix(theta, X, Y):
    z = sigmoid(X.dot(theta))
    return (1/m)*(np.transpose(X) * z * (1-z)).dot(X)

# Implementation of Newton's method.
def newtonsMethod(initialTheta, X, Y, epsilon):
    thetaPrev = initialTheta - 2 * epsilon
    thetaNext = initialTheta  
    thetaArray = []   
    i = 0
    while(lin.norm(thetaPrev - thetaNext) > epsilon):
        thetaArray.append(thetaNext)
        thetaPrev = thetaNext
        thetaNext = thetaPrev - lin.inv(hesssianMatrix(thetaPrev, X, Y)).dot(gradientMatrix(thetaPrev, X, Y))
        i = i + 1 
    print("No. of iterations. = ", i)
    return (thetaNext, np.array(thetaArray))
    
if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        prog='Logistic Regression',
        description='Finding optimal theta using Newton\'s method.')
    argparser.add_argument('parameters', type=float, nargs = 1, help = 'epsilon')
    args = argparser.parse_args()

    X = np.genfromtxt('ass1_data/logisticX.csv', delimiter = ',')
    Y = np.genfromtxt('ass1_data/logisticY.csv', delimiter = ',')
    m = len(X)
    X = np.c_[ np.ones(m),normalize(X[:,0]), normalize(X[:,1])]
    initialTheta = np.zeros(3)
    epsilon = args.parameters[0]
    print("\nInitial theta = ", initialTheta)
    print("Epsilon = ", epsilon)
    (theta,thetaArray) = newtonsMethod(initialTheta, X ,Y, epsilon)
    print("Theta obtained using Newton's Method = ", theta)
    plotLine(X, Y, theta)