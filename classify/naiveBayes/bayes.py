# -*-coding:utf-8-*-

# **************朴素贝叶斯算法****************
import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

# 数据集多个文件的单词汇去重后汇总到list中
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):\
    # 创建长度与vocabLis等长的全0列表
    # 用来记录inputSet中出现且vocabList也有的单词
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word:%s is not in my Vocabulary!' % word)
    return returnVec


# 朴素贝叶斯分类器训练函数
# 计算每个词在各分类下出现的频率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)

    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom

    return p0Vect,p1Vect,pAbusive

if __name__ == '__main__':
    postingList,classVec = loadDataSet()
    # 若不对myVocabList中单词排序, list中单词顺序每次都会不一样
    myVocabList = createVocabList(postingList)
    # hasWordVec = setOfWords2Vec(myVocabList,postingList[0])
    # print(hasWordVec)
    # print(myVocabList[11])

    trainMat = []
    for postingDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postingDoc))

    p0V,p1V,pAb = trainNB0(trainMat,classVec)
    print(p0V)
    print(p1V)
    print(pAb)


