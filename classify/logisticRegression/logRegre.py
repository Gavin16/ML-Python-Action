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
    plotBestFit(weights)
