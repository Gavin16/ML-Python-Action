# -*-coding:utf-8-*-

import numpy as np
import os
import knn
import treePloter
import decisionTree as dtree

# 数据储存路径
trainingDataPath = 'D:\\DataCenter\\document\\ML\\data\\handWritingClassify\\training\\trainingDigits'
testDataPath = 'D:\\DataCenter\\document\\ML\\data\\handWritingClassify\\test\\testDigits'

fileName = '\\0_0.txt'

# 约会  训练数据存储路径
datingDataPath = os.path.dirname(os.path.dirname(os.getcwd())) + '\\data\\datingTestSet2.txt'


# 测试手写数字识别
def testHandWriting():
    vector = knn.image2vector(trainingDataPath + fileName)
    print(vector)
    knn.handWritingClassTest(trainingDataPath, testDataPath)


# 测试决策树分类
def testDecisionTree():
    dataSet, labels = dtree.createDataSet()
    # decisionTree中createTree方法会修改labels的应用 故labels重新拷贝一份作为ctreateTree传参
    labelsCT = labels[:]
    myTree = dtree.createTree(dataSet, labelsCT)
    classResult = treePloter.classify(myTree, labels, [0, 1])
    print(classResult)


# 对隐形眼镜数据集构造决策树
def testLenses():
    # 读取lenses.txt文件
    rootDir = os.path.dirname(os.getcwd())
    print(rootDir)
    dataFullDir = rootDir + '\\data\\ch03\\lenses.txt'
    fr = open(dataFullDir)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    fr.close()
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = dtree.createTree(lenses, lensesLabels)
    treePloter.createPlot(lensesTree)


# 注意路径名不要与模块名同名，否则会报 module 'xxxx' has no attribute 'xxx' 错误
if __name__ == '__main__':
    # testHandWriting()
    # testDecisionTree()
    dataSet, labels = dtree.createDataSet()
    labelsCT = labels[:]
    myTree = dtree.createTree(dataSet, labelsCT)
    dtree.storeTree(myTree, 'decisionTreeStorage.txt')

    myTree2 = dtree.grabTree('decisionTreeStorage.txt')
    print(myTree2)

    testLenses()

