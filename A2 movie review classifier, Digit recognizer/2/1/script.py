import random
import numpy as np
import pickle
from collections import Counter
import argparse
import time
from skimage.io import imread, imsave

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


def testData(testX, trainParameters):
    predictions = []
    for digit in testX:
        predict = Counter()
        for label1 in range(0, 10):
            for label2 in range(label1 + 1 , 10):
                key = str(label1) + str(label2)
                (w,b) = trainParameters[key]
                if(b + np.sum(w * digit) > 0):
                    predict[label2] += 1
                else:
                    predict[label1] += 1
        predictions.append(predict.most_common(1)[0][0])
    return predictions

def inputParsing(pathToFile, typeData):

    data = np.genfromtxt(pathToFile, delimiter = ',', dtype = int)
    m = np.shape(data)[0]
    n = np.shape(data)[1]
    if(typeData == 'testInput'):
        # print("Yoo")
        return data 
    X = data[:,0 : n - 1]
    Y = data[:, n - 1] 
    if(typeData == 'test'):
        return(X.tolist(), Y.tolist())
    trainX = {}
    for i in range(0, 10):
        trainX[i] = []
    for i in range(0, m):
        trainX[Y[i]].append(X[i])
    return trainX


# Bigger Number is y = 1 and smaller is y = -1
def trainData(trainX, C, no_iterations, batch_size):
    data = {}
    for i in range (0 , 10):
        print("i = ",i)
        for j in range (i + 1, 10):
            print("j = ",j)
            X1 = trainX[i] 
            X2 = trainX[j]
            Y1 = -1 * np.ones(len(X1))
            Y2 = np.ones(len(X2))
            X = np.r_[X1, X2]
            Y = np.r_[Y1, Y2]
            key = str(i) + str(j)
            data[key] = (pegasos(X, Y, C, no_iterations, batch_size))
    return data

    

def drawPNG(testY, testX, predictions):
    count = 0
    for i in range(0, len(testX)):
        if predictions[i] != testY[i]:
            print("Actual = ",testY[i])
            print("Predicted = ",predictions[i])
            image_name = str(count) + "Actual = "+ str(testY[i]) + "Predicted = " + str(predictions[i])
            x = np.array(testX[i])
            x = x.reshape(28, 28)
            imsave(image_name + '.png', x)
            count += 1
            if count >= 15:
                break


def pegasos(trainX, trainY, C, no_iterations, k):
    m = len(trainX)
    n = len(trainX[0])
    lambdaa = 1 / (m * C)
    prev_w = np.zeros(n)
    next_w = np.zeros(n)
    prev_b = 0.0
    next_b = 0.0
    for t in range(0, no_iterations):
        prev_w = next_w
        prev_b = next_b
        it = np.random.randint(m, size = k)
        nt = 1 / (lambdaa * (t + 1))
        next_w = (1 - nt * lambdaa) * prev_w
        next_b = prev_b
        for j in range(0, k):
            x = trainX[it[j]]
            y = trainY[it[j]]
            if(y * np.sum(prev_w * x) < 1): 
                next_w += (nt / k) * y * x
                next_b += (nt / k)  
        # print (next_b)
    print(next_b)
    return (next_w, next_b)


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
# no_iterations = 1000
# batch_size = 100
# trainParameters = trainData(trainX, C, no_iterations, batch_size)
# pickleDump('trainParameters', trainParameters)
trainParameters = pickleLoad('trainParameters')
    
# (testX, testY) = inputParsing('mnist/test.csv', 'test')
# (trX, trY) = inputParsing('mnist/train.csv', 'test')
# pickleDump('trX',trX)
# trX = pickleLoad('trX')
# pickleDump('trY',trY)
# trY = pickleLoad('trY')

print("Reading Test Input data.......")
testX = inputParsing(args.paths[0], 'testInput')
# pickleDump('testX',testX)
# testX = pickleLoad('testX')
# pickleDump('testY',testY)
# testY = pickleLoad('testY')
print(len(testX))
# print(len(testY))
# print((testY))
print("Predicting Test digits......")
testPredictions = testData(testX, trainParameters)
# drawPNG(testY, testX, testPredictions)
# trainPredictions = testData(trX, trainParameters)
# print(trainPredictions)
fw =  open(pathToOutputTestFile, "w")
for i in range(0, len(testPredictions)):
    fw.write(str(testPredictions[i]) + "\n")
fw.close()
# print("Test Accuracy = ", calculateAccuracy(testPredictions, testY))
# print("Train Accuracy = ", calculateAccuracy(trainPredictions, trY))
