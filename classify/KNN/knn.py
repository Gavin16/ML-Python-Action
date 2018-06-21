# -*-coding:utf-8-*-

##################### K近邻算法实现 #####################

import numpy as np
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels



# classify0()函数有4个输入参数：
# 用于分类的输入向量是inX，输入的训练样本集为dataSet 标签向量为labels，最后的参数k表示用于选择最近邻居的数目
# 其中标签向量的元素数目和矩阵dataSet的行数相同。程序清单2-1使用欧氏距离公式，计算两个向量点xA和xB之间的距离
def classify0(inx,dataSet,labels,k):
    # shape[0] 获取多少行数据
    dataSetSize = dataSet.shape[0]
    # tile函数将inx填充到dateSet同一维度,inX是单条记录
    diffMat = np.tile(inx,(dataSetSize,1)) - dataSet
    # 计算欧式距离：平方和开根号
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 计算出来的距离升序排序
    sortedDistIndicies = distances.argsort()
    classCount={}

    # 在前k个距离最近的元素中投票找出出现次数最多的标签
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # 对key指定的参数排序,reverse = True则倒序排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# 测试classify0方法
if __name__ == '__main__':
    group,labels = createDataSet()
    result = classify0([0,0],group,labels,2)
    print("预测结果为:"+result)

