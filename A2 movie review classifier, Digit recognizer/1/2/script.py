import numpy as np
import argparse
import re
from collections import Counter
import pickle
import time
import math
import random
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

p_stemmer = PorterStemmer()
ratings = [1, 2, 3, 4, 7, 8, 9, 10]
wordsPerRating = Counter()
vocabulary = set()
trainModel = {}
phi = Counter()

def pickleDump(file, object):
    fileHandle = open(file, "wb")
    pickle.dump(object, fileHandle)

def pickleLoad(file):
    fileHandle = open(file, "rb")
    return pickle.load(fileHandle)

def randomPrediction(testY, n):
    m = len(testY)
    total = 0.0
    for j in range(0,n):
        predictions = []
        for i in range(0, m):
            predictions.append(random.choice(ratings))
        total += calculateAccuracy(predictions, testY)
    total = total / n
    return total

def mostOccuringClassPrediction():
    max = 0
    for label in phi:
        if (phi[label] > max):
            max = phi[label]
    return max * 100

def confusionMatrix(predictions, testY):
    m = len(predictions)
    matrix = np.zeros([11, 11], dtype = 'int')
    for i in range(0, m):
        predicted = predictions[i]
        actual = testY[i]
        matrix[actual][predicted] += 1
    matrix = np.delete(matrix, [0,5,6], 0)    
    matrix = np.delete(matrix, [0,5,6], 1)    
    matrix = matrix * 100 / m
    actualNames = list(map(lambda  x: "Actual: " + str(x), ratings))
    predictionNames = list(map(lambda  x: "Prediction: " + str(x), ratings))
    df = pd.DataFrame(matrix, index=actualNames, columns=predictionNames)
    df.to_csv('confusionMatrix.csv', index=True, header=True, sep=',')

def removeHtmlTags(str):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', str)
  return cleantext

def removePunctuations(str):
    punc = ['.', '\'', '?', ',', '"', '!', '(', ')', ':', ';', '/' ]
    n = len(str)
    ans = ""
    for i in range(0, n):
        if str[i] not in punc:
            ans = ans + str[i]
        else:
            ans = ans + " "
    return ans

def processInput(input):
    trainX = []    
    n = len(input)
    for i in range(0,n):
        doc = input[i]
        doc = removeHtmlTags(doc)
        doc = removePunctuations(doc)
        doc = doc.lower()
        doc = doc.split()
        doc = list(filter(lambda x: x not in en_stop, doc))
        doc = [p_stemmer.stem(token) for token in doc]
        doc = Counter(doc)
        trainX.append(doc)
    return trainX

def buildVocab(trainX):
    for document in trainX:
        for word in document:
            vocabulary.add(word)

def trainData(trainX, trainY):
    m = len(trainY)

    trainModel = {}
    for label in ratings:
        trainModel[label] = Counter() 

    for i in range(0, m):
        label = trainY[i]
        trainIth = trainX[i]
        for word in trainIth:
            if not word in trainModel[label]:
                trainModel[label][word] = 0
            trainModel[label][word] += 1
            wordsPerRating[label] += 1

    return trainModel

def calculateProbability(label, testIth, laplace):
    vocabSize = len(vocabulary)
    ans = math.log(phi[label])
    frequencyPerClass = trainModel[label]
    totalWordsInClass =  math.log(wordsPerRating[label] + (laplace * vocabSize))
    for word in testIth.keys():
        if word in vocabulary:
            occInLabel = frequencyPerClass[word]
            ans += math.log(laplace + occInLabel)
            ans -= totalWordsInClass
    return ans

def testData(trainModel, phi, testX, laplace):
    predictions = []

    for testIth in testX:
        maxProb = float("-inf")
        index = -1
        for label in phi.keys():
            # print(label)
            prob = calculateProbability(label, testIth, laplace )
            if(prob > maxProb):
                maxProb = prob
                index = label
        predictions.append(int(index))
    return predictions

def calculateAccuracy(predictions, actual):
    accuracy = 0
    for i in range(0, len(actual)):
        if(actual[i] == predictions[i]):
            accuracy += 1
    return (accuracy * 100 / len(actual))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog='Naive Bayes',
        description='Predicting labels using Naive Bayes')
    argparser.add_argument('paths', type=str, nargs = 2, help = 'pathToInputTestFile, pathToOutputTestFile')
    args = argparser.parse_args()

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # pathToTrainX = "imdb/imdb_train_text.txt"
    # pathToTrainY = "imdb/imdb_train_labels.txt"
    pathToTestX = args.paths[0]
    pathToTestPrediction = args.paths[1]

    # pickleDump('en_stop', en_stop)
    en_stop = pickleLoad('en_stop')
    # print("Reading training input files........")
    # with open(pathToTrainX) as f:
    #     trainX = f.readlines()
    # with open(pathToTrainY) as f:
    #     trainY = f.readlines()
    # trainY = [int(i) for i in trainY]

    # t0 = time.time()
    # print("Processing training reviews........")
    # trainX = processInput(trainX)
    # t1 = time.time()
    # print("Processing Input Time = ", t1 - t0)

    # print("Building Vocabulary........")

    # buildVocab(trainX)

    # pickleDump('vocabulary', vocabulary)
    vocabulary = pickleLoad('vocabulary')

    # m = len(trainY)
    # phi = Counter()
    # for label in trainY:
    #     phi[label] += 1
    # phi = dict(phi)
    # for label in phi:
    #     phi[label] = phi[label] / m

    # pickleDump('phi', phi)
    phi = pickleLoad('phi')

    # print("Training input data........")
    # t0 = time.time()
    # trainModel = trainData(trainX, trainY)
    # t1 = time.time()
    # print("Training time = ", t1 - t0)
    # pickleDump('trainModel', trainModel)
    trainModel = pickleLoad('trainModel')

    # pickleDump('wordsPerRating', wordsPerRating)
    wordsPerRating = pickleLoad('wordsPerRating')


    print("Reading test reviews file........")
    with open(pathToTestX) as f:
        testX = f.readlines()

    laplace = 1

    print("Processing test reviews........")
    t0 = time.time()
    testX = processInput(testX)
    t1 = time.time()
    print("Processing Input Time = ", t1 - t0)

    # print("Calculating training set Accuracy........")
    # t0 = time.time()
    # trainPredictions = testData(trainModel, phi, trainX, laplace)
    # t1 = time.time()
    # print("Prediction time = ", t1 - t0)
    # print("Training accuracy = ", calculateAccuracy(trainPredictions, trainY))


    print("Calculating test set Accuracy........")
    t0 = time.time()
    testPredictions = testData(trainModel, phi, testX, laplace)
    t1 = time.time()
    print("Prediction time = ", t1 - t0)

    print("Writing Predictions to Output File")
    f = open(pathToTestPrediction, 'w')
    for p in testPredictions:
        f.write(str(p)+'\n')
    f.close()

    # print("Reading test labels file........")
    # pathToTestY = "imdb/imdb_test_labels.txt"
    # with open(pathToTestY) as f:
    #     testY = f.readlines()
    # testY = [int(i) for i in testY]

    # print("Test accuracy = ", calculateAccuracy(testPredictions, testY))
    # print("Random guess prediction accuracy = ", randomPrediction(testX, 100))
    # print("Most occuring class prediction accuracy = ", mostOccuringClassPrediction())

    # print("Creating Confusion Matrix")
    # confusionMatrix(predictions, testY)

