import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def normalize(X):
    mu = np.average(X) 
    sigma = np.var(X)
    return (X - mu ) / np.sqrt(sigma)

def plotData(X , Y):
    print("\nPlotting input data. ")
    plt.plot(X,Y,'r+')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show(block = False)
    input("Press Enter to continue.")

def plotCurve(X, Y, tau):
    plotData(X[:,1],Y)
    print("\nPlotting Straight Line for unweighted linear regression using theta from Normal Equations. ")
    W = np.eye(m)
    theta = normalEquationTheta(X ,Y, W)
    Y1 = X.dot(theta)
    plt.plot(X[:,1] , Y1, 'b')
    plt.draw()
    input("Press Enter to continue.")
    print("\nPlotting weighted regression curve.")
    X2 = np.arange(-2, 2, 0.02)
    for x in X2:
        W = weights(x, X, tau)
        theta = normalEquationTheta(X ,Y, W)
        y = np.array([1 , x]).dot(theta)
        plt.plot([x], [y], marker = 'o', markerfacecolor = 'green', markeredgecolor = 'green', markersize = '1.5')
    plt.draw()
    input("Press Enter to continue.")  
    plt.close()

# Returns value of theta obtained using Normal Equations
def normalEquationTheta(X, Y, W):
    return lin.inv(np.transpose(X).dot(W).dot(X)).dot(np.transpose(X)).dot(W).dot(Y)

# Returns a diagonal weight matrix for weighted linear regression.
# x - test input vector 
# X - Training data
# tau - Weight parameter.
def weights(x, X, tau):
    X = np.square(lin.norm((x - X), axis = 1))
    return np.diag(np.exp( X / (-2 * tau * tau)))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        prog='Weighted Linear Regression',
        description='Finding optimal theta using normal equations.')
    argparser.add_argument('parameters', type=float, nargs = 1, help = 'tau')
    args = argparser.parse_args()

    X = np.genfromtxt('ass1_data/weightedX.csv', delimiter = ',')
    X = normalize(X)
    Y = np.genfromtxt('ass1_data/weightedY.csv', delimiter = ',')
    m = len(X)
    X = np.c_[np.ones(m) , X]
    tau = args.parameters[0]
    print("Tau = ", tau)
    plotCurve(X, Y, tau)

    


