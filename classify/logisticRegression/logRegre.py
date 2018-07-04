# -*-coding:utf-8-*-

# **************************逻辑斯蒂回归****************************
import numpy as np
import os


# 测试数据集读入内存
def loadDataSet():
    dataMat = [];
    labelMat = []
    moduleDir = os.path.dirname(os.getcwd());
    projDir = os.path.dirname(moduleDir)
    fr = open(projDir + '\\data\\ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# 阶跃函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# 梯度上升算法;统一使用numpy的Matrix,Vector
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix.transpose())
    alpha = 0.001  # 步长设置
    maxCycles = 500  # 最大循环次数
    weights = np.ones((m, 1))  # 初始权重

    # 迭代获取权重值，调整方式为：初试权值对结果的估计的偏差乘以步长乘以元素自身 构成反馈加到原权值上
    # 原公式为：w = w + alpha*delt(f) ; 这里为什么可以直接乘以元素自身
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        delta = alpha * error * np.array(dataMatrix[i])
        weights = weights + delta
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=500):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * np.array(dataMatrix[randIndex])
            del dataIndex[randIndex]
    return weights

#
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0

# 从疝气病症预测病马的死亡率
def colicTest():
    import os
    folderDir = os.path.dirname(os.getcwd())
    rootDir = os.path.dirname(folderDir)
    frTrain = open(rootDir + '\\data\\ch05\\horseColicTraining.txt')
    frTest = open(rootDir + '\\data\\ch05\\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))

        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is :%f' % errorRate)
    return errorRate

# 多次测试
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' %(numTests,errorSum/float(numTests)))


# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)

    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')

    x = np.arange(-3,3,0.1).reshape(1,60)
    print('shape of X:',np.shape(x))
    y = (-weights[0] - weights[1]*x)/weights[2]
    print('shape of Y:',np.shape(y))
    print('values of Y:',y)
    # ax.plot(x,y,'k--',linewidth=2,markersize=3)
    ax.plot(x, y, color='black', marker='*', linestyle='solid',
                linewidth=1, markersize=2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    print(weights)
    # plotBestFit(weights)
    # weights1 = stocGradAscent0(dataMat,labelMat)
    # plotBestFit(weights1)
    # weights2 = stocGradAscent1(dataMat,labelMat)
    # plotBestFit(weights2)
    multiTest()