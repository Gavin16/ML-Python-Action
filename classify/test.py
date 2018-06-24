# -*-coding:utf-8-*-

import numpy as np
import os
import knn

# 数据储存路径
trainingDataPath = 'D:\\DataCenter\\document\\ML\\data\\handWritingClassify\\training\\trainingDigits'
testDataPath = 'D:\\DataCenter\\document\\ML\\data\\handWritingClassify\\test\\testDigits'

fileName = '\\0_0.txt'

# 约会  训练数据存储路径
datingDataPath = os.path.dirname(os.path.dirname(os.getcwd())) + '\\data\\datingTestSet2.txt'


# 测试来自各算法的模型
def testHandWriting():
	vector = knn.image2vector(trainingDataPath + fileName)
	print(vector)
	knn.handWritingClassTest(trainingDataPath,testDataPath)

if __name__ == '__main__':
	testHandWriting()



