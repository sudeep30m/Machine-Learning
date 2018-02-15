import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from matplotlib import cm

# Normalizing 1D array X
def normalize(X):
    mu = np.average(X) 
    sigma = np.var(X)
    return (X - mu ) / np.sqrt(sigma)

# Plotting input data
def plotData(X , Y ):
    print("\nPlotting input data. ")
    plt.plot(X,Y,'r+')
    plt.xlabel('Acidity')
    plt.ylabel('Density')
    plt.show(block = False)
    input("Press Enter to continue.")
    # plt.close()

# Function for plotting linear hypothesis obtained using gradient descent.
def plotLine(X, Y, theta ):
    plotData(X[:,1],Y)
    print("\nPlotting Straight Line using theta obtained from gradient descent. ")
    Y1 = X.dot(theta)
    plt.plot(X[:,1], Y1, 'b')
    plt.draw()
    input("Press Enter to continue.")
    plt.close()

# Finding optimal theta using gradient descent.
def gradientDescent(alpha, initialTheta, epsilon, X, Y):
    print("\nImplementing Gradient Descent.\n")
    theta = initialTheta
    costPrev = cost(theta, X, Y)
    costNext = costPrev - 2 * epsilon 
    thetaArray = []   
    i = 0
    while(costPrev - costNext > epsilon):
        thetaArray.append(theta)
        costPrev = cost(theta, X ,Y)
        theta = theta - alpha * costDerivative(theta, X, Y)
        costNext = cost(theta, X, Y)
        i = i + 1
    print("No. of iterations. = ", i)
    return (theta,np.array(thetaArray))

# Function for plotting 3d meshgrid Z = cost X = theta0 Y = theta1
def plotCostFunctionMesh(thetaArray, theta, X, Y ):
    print("\nPlotting 3D Mesh.")
    theta0Array = np.linspace(-1,3,500)
    theta1Array = np.linspace(-2,2,500)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T0, T1 = np.meshgrid(theta0Array, theta1Array)
    zs = np.array([cost(np.array([x, y]), X, Y) for x,y in zip(np.ravel(T0), np.ravel(T1))])
    Z = zs.reshape(T0.shape)
    mesh = ax.plot_surface(T0, T1, Z, cmap = cm.winter,
                       linewidth=0, antialiased=False)
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cost function')
    fig.colorbar(mesh, shrink=0.5, aspect=5)
    plt.show(block = False)
    input("Press Enter to continue.")
    for i in range(0, len(thetaArray)):
        # print(i)
        plt.plot([thetaArray[i,0]], [thetaArray[i,1]], cost(thetaArray[i], X, Y), marker = 'o', markerfacecolor = 'black', markeredgecolor = 'black', markersize = '2')
        plt.draw()
        plt.pause(0.01)
    input("Press Enter to continue.")
    plt.close()

# Function for plotting Contours of cost function.
def plotContour(thetaArray, theta, X, Y ):
    print("\nPlotting contours.\n")
    theta0Array = np.linspace(-1,3,500)
    theta1Array = np.linspace(-2,2,500)
    T0, T1 = np.meshgrid(theta0Array, theta1Array)
    zs = np.array([cost(np.array([x,y]), X, Y) for x,y in zip(np.ravel(T0), np.ravel(T1))])
    Z = zs.reshape(T0.shape)
    fig = plt.figure()
    CS = plt.contour(T0, T1, Z)
    plt.plot([theta[0]], [theta[1]], marker = 'x', markerfacecolor = 'red', markeredgecolor = 'red', markersize = '4')

    plt.clabel(CS, inline=1, fontsize=10)
    plt.ion()
    plt.show(block = False)

    input("Press Enter to continue.")
    for i in range(0, len(thetaArray)):
        plt.plot([thetaArray[i,0]], [thetaArray[i,1]], marker = 'o', markerfacecolor = 'blue', markeredgecolor = 'blue', markersize = '1.5')
        plt.draw()
        plt.pause(0.02)
    input("Press Enter to continue.")
    plt.close()

# X is (m, n) list Y is (m) theta is (n) 
def cost(theta, X, Y):
    return (1/2) * ( np.sum( np.square(Y - X.dot(theta) ) ) )

# Cost derivative for linear regression used in gradient descent.
def costDerivative(theta, X, Y):
    return np.transpose(X).dot(X.dot(theta) - Y)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog='Linear Regression',
        description='Finding optimal theta using batch gradient descent')
    argparser.add_argument('parameters', type=float, nargs = 2, help = 'alpha, epsilon')
    args = argparser.parse_args()
    
    X = np.genfromtxt('ass1_data/linearX.csv', delimiter = ',')
    Y = np.genfromtxt('ass1_data/linearY.csv', delimiter = ',')
    m = len(X)
    X = normalize(X)
    X = np.c_[np.ones(m) , X]
    initialTheta = np.array([-1,2])
    alpha = args.parameters[0]
    epsilon = args.parameters[1]
    print("\nInitial theta = ", initialTheta)
    print("Alpha = ", alpha)
    print("Epsilon = ", epsilon)
    (theta,thetaArray) = gradientDescent(alpha, initialTheta, epsilon, X, Y)
    print("Theta obtained using gradient descent = ", theta)
    plotLine(X, Y, theta)
    plotCostFunctionMesh(thetaArray, theta, X, Y)
    plotContour(thetaArray, theta, X, Y)
