#-*-coding:utf-8-*-
from numpy import *

inx = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
# print (inx.shape[0])

dataSet = [[1,0],[2,1],[1,1],[2,1],[2,2],[1,2],[2,2]]
# print(dataSet.__len__())
padedInx = tile(inx,(dataSet.__len__(),1))

print(padedInx)