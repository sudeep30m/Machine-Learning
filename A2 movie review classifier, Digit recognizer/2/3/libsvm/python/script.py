import random
import numpy as np
import pickle
from collections import Counter
from svmutil import *
import argparse
import time
from skimage.io import imread, imsave
import pandas as pd

def pickleDump(file, object):
    fileHandle = open(file, "wb")
    pickle.dump(object, fileHandle)

def pickleLoad(file):
    fileHandle = open(file, "rb")
    return pickle.load(fileHandle)

def confusionMatrix(predictions, testY):
    m = len(predictions)
    matrix = np.zeros([10, 10], dtype = 'int')
    for i in range(0, m):
        predicted = int(predictions[i])
        actual = int(testY[i])
        matrix[actual][predicted] += 1
    matrix = matrix * 100 / m
    ratings = ['0','1','2','3','4','5','6','7','8','9']
    actualNames = list(map(lambda  x: "Actual: " + str(x), ratings))
    predictionNames = list(map(lambda  x: "Prediction: " + str(x), ratings))
    df = pd.DataFrame(matrix, index=actualNames, columns=predictionNames)
    df.to_csv('confusionMatrix.csv', index=True, header=True, sep=',')

def calculateAccuracy(predictions, actual):
    accuracy = 0
    for i in range(0, len(actual)):
        if(actual[i] == predictions[i]):
            accuracy += 1
    return (accuracy * 100 / len(actual))

def inputParsing(pathToFile, typeData):

    data = np.genfromtxt(pathToFile, delimiter = ',', dtype = int)
    m = np.shape(data)[0]
    n = np.shape(data)[1]
    if(typeData == 'testInput'):
        data = data / 255
        return data.tolist() 
    
    X = data[:,0 : n - 1]
    Y = data[:, n - 1] 

    if(typeData == 'test'):
        X = X / 255
        return(X.tolist(), Y.tolist())
    trainX = {}
    for i in range(0, 10):
        trainX[i] = []
    for i in range(0, m):
        trainX[Y[i]].append(X[i])
    return trainX


# trainX is list of lists trainY is list
# def libsvm(trainX, trainY, testX, testY, gamma, c):
def libsvm(testX, gamma, c):
    # Linear Kernel 
    # param = '-g ' + str(gamma) + ' -c ' + str(c) + ' -t 0' 
    # m = svm_train(trainY, trainX, param)
    # svm_save_model("c = "+ str(c) + '(linear).model', m) 

    # print('Linear Train Accuracy')
    # p1,p2,p3 = svm_predict(trainY, trainX, m)

    # print('Linear Test Accuracy')
    # p1,p2,p3 = svm_predict(testY, testX, m)

    m = svm_load_model('c = 5.0(gaussian).model')
    testY = [0] * len(testX) 
    p1,p2,p3 = svm_predict(testY, testX, m)
    return p1


argparser = argparse.ArgumentParser(
    prog='SVM',
    description='Predicting digits using SVM')
argparser.add_argument('paths', type = str, nargs = 2, help = 'pathToInputTestFile, outputFileForPredicitons')
# argparser.add_argument('parameter', type = float, nargs = 1, help = 'c')
args = argparser.parse_args()
pathToInputTestFile = args.paths[0]
pathToOutputTestFile = args.paths[1]
    
# (testX, testY) = inputParsing('mnist/test.csv', 'test')
# (trX, trY) = inputParsing('mnist/train.csv', 'test')
# trX = trX / 255
# testX = testX / 255
# pickleDump('trX',trX)
# trX = pickleLoad('trX')
# pickleDump('trY',trY)
# trY = pickleLoad('trY')

testX = inputParsing(pathToInputTestFile, 'testInput')

# pickleDump('testX',testX)
# testX = pickleLoad('testX')
# pickleDump('testY',testY)
# testY = pickleLoad('testY')
# print(len(testX))
# print(len(testY))
# print((testY))
print("Predicting Data")
# predictions = testData(testX, trainParameters)
# print(len(predictions))
# print("Test Accuracy = ", calculateAccuracy(predictions, testY))
gamma = 0.05
c = 5.0
print("c = ", c)
print("LIBSVM")
testPredictions = libsvm(testX, gamma, c)
# pickleDump('testPredictions', testPredictions)
# testPredictions = pickleLoad('testPredictions')

# print("Accuracy = ",calculateAccuracy(testPredictions, testY))
fw =  open(pathToOutputTestFile, "w")
for i in range(0, len(testPredictions)):
    fw.write(str(int(testPredictions[i])) + "\n")
fw.close()

# libsvm(trX, trY, testX, testY, gamma, c)
# for c in [0.00001, 0.001, 1, 5, 10]:

# param = '-g ' + str(gamma) + ' -c ' + str(c) + ' -h 0 -t 2 -v 10' 
# m = svm_train(trY, trX, param)



# print(np.random.randint(2))