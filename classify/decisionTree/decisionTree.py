#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 决策树算法实现分：
# (1)最优特征选取
# (2)决策树生成 -- ID3算法, C4.5算法
# (3)决策树枝减
# 三步

import math as math
import operator as oprt


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
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# 简单训练集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 选出属性下标=axis 对应值为 value的数据子集(子集中去掉当前属性(下标为axis的属性))
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
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
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 计算该属性确定后的条件信息熵
            newEntropy += prob * calcEnt(subDataSet)
        # i号属性对应的信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 统计数据集中分类标签每个标签出现的次数,返回次数最多的标签
def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    storedClassCount = sorted(classCount.items(), key=oprt.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果类标签只有一个 返回该类的样例数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集中只有一个特征,则无需做特征选择，直接选择出现次数最多的类作为分类结果
    if len(dataSet[0]) == 1:
        return majorityCount(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 取出当前最优特征的所有可能取值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制所有属性名到另一个list用来迭代
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    dataEnt = calcEnt(dataSet)
    print("计算得到数据集的经验熵为：%f" % dataEnt)

    retDataSet = splitDataSet(dataSet, 1, 1)
    print(retDataSet)
    chooseBestFeatureToSplit(dataSet)
    desTree = createTree(dataSet, labels)
    print(desTree)
