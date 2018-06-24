#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 决策树算法实现分：
# (1)最优特征选取
# (2)决策树生成 -- ID3算法, C4.5算法
# (3)决策树枝减
# 三步

import math as math

# 计算数据集的经验熵(香农熵/信息量)
def calcEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	# 数据集中所有标记出现的概率与log(p,2)的乘积和
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob*math.log(prob,2)
	return shannonEnt


# 简单训练集
def createDataSet():
	dataSet = [[1,1,'yes'],
				[1,1,'yes'],
			   [1,0,'no'],
			   [0,1,'no'],
			   [0,1,'no']]
	labels = ['no surfacing','flippers']
	return dataSet,labels


# 选出属性下标=axis 对应值为 value的数据子集(子集中去掉当前属性(下标为axis的属性))
def splitDataSet(dataSet,axis,value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


# 计算返回信息增益最大的属性的id
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcEnt(dataSet)
	bestInfoGain = 0.0
	# 若属性没有信息增益,默认返回-1
	bestFeature = -1
	for i in range(numFeatures):
		# python 列表推导式获取数据集中所有样例的第i个元素，构成列表
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet)/float(len(dataSet))
			# 计算该属性确定后的条件信息熵
			newEntropy += prob*calcEnt(subDataSet)
		# i号属性对应的信息增益
		infoGain = baseEntropy - newEntropy
		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature





if __name__ == '__main__':
	dataSet,labels = createDataSet()
	dataEnt = calcEnt(dataSet)
	print("计算得到数据集的经验熵为：%f" % dataEnt)

	retDataSet = splitDataSet(dataSet,1,1)
	print(retDataSet)
	chooseBestFeatureToSplit(dataSet)


