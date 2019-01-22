import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords


def loadData(filename):
	## load train data & labels, test data & labels
	lines = [line.rstrip('\n') for line in open(filename)]
	Dfile = [i.split() for i in lines]
	return Dfile

def loadLabels(filename):
	Lfile = []
	with open(filename) as f:
		for line in f:
		    Lfile.extend(line.strip().split(' '))
	return Lfile

def rmvStopwords_train(file):
	stop_words = list(stopwords.words('english'))
	wordsList = []
	[wordsList.extend(x) for x in file]
	DataNoStpwrd = [x for x in wordsList if x not in stop_words ]
	return DataNoStpwrd

def rmvStopwords(file):
	stop_words = list(stopwords.words('english'))
	DataNoStpwrd = []
	for i in file:
		row = [x for x in i if x not in stop_words]
		DataNoStpwrd.append(row)
	return DataNoStpwrd

def separateWords(dataList, labelList):
	# find unique vocablabies in the training set
	sents = []
	[sents.extend(x) for x in dataList]
	trNoStpwrds = rmvStopwords(sents)
	uniqueVoc = set(sents)
	print('total number of uniqueVoc: ', len(uniqueVoc))

	# merge two list and separate the data into '1' and '0'
	oneData = [x for x, y in zip(dataList, labelList) if y=='1']
	zeroData = [x for x, y in zip(dataList, labelList) if y=='0']

	# apply stopwords
	oneTrData = rmvStopwords_train(oneData)
	zeroTrData = rmvStopwords_train(zeroData)

	# remove duplicates and find unique words in the train documents classified as '1' and '0'
	uniqueOne = set(oneTrData)
	uniqueZero = set(zeroTrData)

	# find words missing in each one and zero list compared to all unique vocablaries
	wordsNotInOne = [x for x in uniqueVoc if x not in uniqueOne]
	wordsNotInZero = [x for x in uniqueVoc if x not in uniqueZero]

	# count frequency of each words
	wordCountOne = Counter(oneTrData )
	wordCountZero = Counter(zeroTrData)
	for i in wordsNotInOne:
		wordCountOne[i] = 0
	for i in wordsNotInZero:
		wordCountZero[i] = 0

	return uniqueVoc, oneTrData, zeroTrData, wordCountOne, wordCountZero

def addProb(uniqueVoc, wordsCountOne, wordsCountZero, oneTrData, zeroTrData):
	numWordsOne = len(oneTrData)
	numWordsZero = len(zeroTrData)

	wordcountONEDic = {}
	for key, item in wordsCountOne.items():
		probWord = (item+1)/(numWordsOne+len(uniqueVoc))
		wordcountONEDic.setdefault(key,[item]).append(probWord)

	wordcountZERODic = {}
	for key, item in wordsCountZero.items():
		probWord = (item+1)/(numWordsZero+len(uniqueVoc))
		wordcountZERODic.setdefault(key,[item]).append(probWord)

	return wordcountONEDic, wordcountZERODic

def calTotalProb(sentence, priorOnes, priorZeros, wordcountONEDic, wordcountZERODic, uniqueVoc):
	probOne = priorOnes
	probZero = priorZeros
	for word in sentence:
		if word in uniqueVoc:
			probOne *=wordcountONEDic[word][1]
			probZero *=wordcountZERODic[word][1]
		else:
			continue

	return probOne, probZero

def predict(list):
	predList = []
	for i in list:
		if i[0]>i[1]:
			predList.append(1)
		else:
			predList.append(0)
	return predList

def accuracy(list1, list2):
	list2 = list(map(int, list2))
	acc = [1 if i==j else 0 for i, j in zip(list1,list2)]
	print(acc)
	countAcc = sum(acc)
	accRate = countAcc/len(acc)

	return accRate

if __name__ == "__main__":

	# load data, label files
	trData = loadData('traindata.txt')
	trLabels = loadLabels('trainlabels.txt')
	teData = loadData('testdata.txt')
	teLabels = loadLabels('testlabels.txt')

	# count frequency of '1' and '0'
	totalL = len(trLabels)
	print('total: training documents(rows) ', totalL)
	totalOnes = trLabels.count('1')
	totalZeros = trLabels.count('0')

	labelCount = {'totalDocs': totalL, 'classOnes':totalOnes, 'classZeros':totalZeros}
	print(labelCount)

	# separate words for one and zero
	uniqueVoc, oneTrData, zeroTrData, wordsCountOne, wordsCountZero = separateWords(trData, trLabels)
	print('total number of words in "1: ', len(oneTrData))
	print('total number of words in "0": ', len(zeroTrData))

	# calculate priors: P(one), P(zero)=P(-one)
	priorOnes = labelCount['classOnes']/labelCount['totalDocs']
	priorZeros = labelCount['classZeros']/labelCount['totalDocs']
	print('Prior - P(ones) & P(zeros): ', priorOnes, '  ', priorZeros) 

	# add probability for each word to the wordCountOne/Zero dictionary
	wordcountONEDic, wordcountZERODic = addProb(uniqueVoc, wordsCountOne, wordsCountZero, oneTrData, zeroTrData)

	# apply Naive Bays to the test set
	teDataNoStpwrds = rmvStopwords(teData)
	predictOneZero = []
	for sentence in teDataNoStpwrds:
		probOne, probZero = calTotalProb(sentence, priorOnes, priorZeros, wordcountONEDic, wordcountZERODic, uniqueVoc)
		predictOneZero.append([probOne, probZero])

	# predict the class
	predList = predict(predictOneZero)
	print(predList)

	# calculate accuracy rate
	accRate = accuracy(predList, teLabels)
	print('Accuracy Rate: {:.2f}%'.format(accRate*100))




