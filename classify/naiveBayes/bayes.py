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


def setOfWords2Vec(vocabList, inputSet): \
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
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 贝叶斯估计, 引入拉普拉斯平滑
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)
    p0Denom = 1.0;
    p1Denom = 1.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postingDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classifyed as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 从文本中解析出单词, 以所有不是数字或者字母的字符划分，过滤掉长度不超过2的字符串
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamText():
    docList = [];classList = [];fullText = []
    # 文件的根路径
    rootDir = os.path.dirname(os.getcwd())
    for i in range(1,26):
        wordList = textParse(open(rootDir+'\\data\\ch04\\spam\\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(rootDir+'\\data\\cho4\\ham\\%d.txt' % i).read())

        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50);textSet = []

    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []

    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:',float(errorCount)/len(testSet))



if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    # 若不对myVocabList中单词排序, list中单词顺序每次都会不一样
    myVocabList = createVocabList(postingList)
    # hasWordVec = setOfWords2Vec(myVocabList,postingList[0])
    # print(hasWordVec)
    # print(myVocabList[11])

    trainMat = []
    for postingDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))

    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    testingNB()
