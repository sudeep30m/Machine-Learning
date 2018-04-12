import random
import numpy as np
import pickle
from collections import Counter
from svmutil import *
import argparse
import time

def pickleDump(file, object):
    fileHandle = open(file, "wb")
    pickle.dump(object, fileHandle)

def pickleLoad(file):
    fileHandle = open(file, "rb")
    return pickle.load(fileHandle)

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
        # print("Yoo")
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
def libsvm(testX, gamma, c):
# def libsvm(trainX, trainY,testX, testY, gamma, c):


    # Linear Kernel
    # param = '-g ' + str(gamma) + ' -c ' + str(c) + ' -t 0' 
    # m = svm_train(trainY, trainX, param)
    # svm_save_model("c = "+ str(c) + '(linear).model', m) 

    # print('Linear Train Accuracy')
    # p1,p2,p3 = svm_predict(trainY, trainX, m)

    # print('Linear Test Accuracy')
    # p1,p2,p3 = svm_predict(testY, testX, m)

    m = svm_load_model('c = 1.0(linear).model')
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

# trainX = inputParsing('mnist/train.csv', 'train')
# pickleDump('trainX', trainX)
# trainX = pickleLoad('trainX')
# C = 1.0
# no_iterations = 500
# batch_size = 100
# trainParameters = trainData(trainX, C, no_iterations, batch_size)
# pickleDump('trainParameters', trainParameters)
# trainParameters = pickleLoad('trainParameters')
# print((trainParameters[0][0]))
    
# (testX, testY) = inputParsing('mnist/test.csv', 'test')
# (trX, trY) = inputParsing('mnist/train.csv', 'test')
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
c = 1.0
# print("c = ", c)
print("LIBSVM")
# libsvm(trX, trY, testX, testY, gamma, c)
testPredictions = libsvm(testX, gamma, c)
fw =  open(pathToOutputTestFile, "w")
for i in range(0, len(testPredictions)):
    fw.write(str(testPredictions[i]) + "\n")
fw.close()

# for c in [0.00001, 0.001, 1, 5, 10]:

# param = '-g ' + str(gamma) + ' -c ' + str(c) + ' -h 0 -t 2 -v 10' 
# m = svm_train(trY, trX, param)



# print(np.random.randint(2))