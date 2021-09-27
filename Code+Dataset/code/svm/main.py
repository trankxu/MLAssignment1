from __future__ import division
import numpy as np
from svm import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
if sys.version_info[0] >= 3:
    xrange = range
# coding:utf-8
"""

参考文献：
[1] 支持向量机训练算法实现及其改进[D].南京理工大学,2005,26-33.
[2] v_JULY_v.支持向量机通俗导论（理解SVM的三层境界）[DB/OL].http://blog.csdn.net/v_july_v/article/details/7624837,2012-06-01.
[3] zouxy09.机器学习算法与Python实践之（四）支持向量机（SVM）实现[DB/OL].http://blog.csdn.net/zouxy09/article/details/17292011,2013-12-13.
[4] JerryLead.支持向量机（五）SMO算法[DB/OL].http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html,2013-12-13.
[5] 黄啸. 支持向量机核函数的研究［D].苏州大学,2008.
"""

import numpy as np
import random
import copy
import math
import time
import sys

if sys.version_info[0] >= 3:
    xrange = range

def calcuKernelValue(train_x, sample_x, kernelOpt = ("linear", 0)):
    kernelType = kernelOpt[0]
    kernelPara = kernelOpt[1]
    numSamples = np.shape(train_x)[0]
    kernelValue = np.mat(np.zeros((numSamples, 1)))
    if kernelType == "linear":
        kernelValue = train_x * sample_x.T
    elif kernelOpt[0] == "rbf":
        sigma = kernelPara
        for i in xrange(numSamples):
            diff = train_x[i, :] - sample_x
            kernelValue[i] = math.exp(diff * diff.T / (-2 * sigma ** 2))
    else:
        print("The kernel is not supported")
    return kernelValue

# 核函数求内积
def calcKernelMat(train_x, kernelOpt):
    numSamples = np.shape(train_x)[0]
    kernealMat = np.mat(np.zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernealMat[:, i] = calcuKernelValue(train_x, train_x[i], kernelOpt)
    return kernealMat

# SVM参数
class svmSruct(object):
    def __init__(self, trainX, trainY, c, tolerance, maxIteration, kernelOption):
        self.train_x = trainX
        self.train_y = trainY
        self.C = c
        self.toler = tolerance
        self.maxIter = maxIteration
        self.numSamples = np.shape(trainX)[0]
        self.alphas = np.mat(np.zeros((self.numSamples, 1)))
        self.b = 0
        self.errorCache = np.mat(np.zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMat(self.train_x, self.kernelOpt)

def calcError(svm, alpha_i):
    func_i = np.multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_i] + svm.b
    erro_i = func_i - svm.train_y[alpha_i]
    return erro_i

def updateError(svm, alpha_j):
    error = calcError(svm, alpha_j)
    svm.errorCache[alpha_j] = [1, error]

# 选取一对 alpha_i 和 alpha_j，使用启发式方法
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]
    alpha_index = np.nonzero(svm.errorCache[:, 0])[0]
    maxstep = float("-inf")
    alpha_j, error_j = 0, 0
    if len(alpha_index) > 1:
        # 遍历选择最大化 |error_i - error_j| 的 alpha_j
        for alpha_k in alpha_index:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_i - error_k) > maxstep:
                maxstep = abs(error_i - error_k)
                alpha_j = alpha_k
                error_j = error_k
    else:
        # 最后一个样本，与之配对的 alpha_j采用随机选择
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = random.randint(0, svm.numSamples - 1)
        error_j = calcError(svm, alpha_j)
    return alpha_j, error_j

# 内循环
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)
    error_i_ago = copy.deepcopy(error_i)
    if (svm.train_y[alpha_i] * error_i < -svm.toler and svm.alphas[alpha_i] < svm.C) or \
        (svm.train_y[alpha_i] * error_i > svm.toler and svm.alphas[alpha_i] > 0):
        # 选择aplha_j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_ago = copy.deepcopy(svm.alphas[alpha_i])
        alpha_j_ago = copy.deepcopy(svm.alphas[alpha_j])
        error_j_ago = copy.deepcopy(error_j)
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - \
                svm.kernelMat[alpha_j, alpha_j]

        # 更新aplha_j, alpha_i
        svm.alphas[alpha_j] = alpha_j_ago - svm.train_y[alpha_j] * (error_i - error_j) / eta
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        elif svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L
        svm.alphas[alpha_i] = alpha_i_ago + svm.train_y[alpha_i] * svm.train_y[alpha_j] * \
                                            (alpha_j_ago - svm.alphas[alpha_j])
        # 问题：为什么只判断alpha_j?
        if abs(alpha_j_ago - svm.alphas[alpha_j]) < 10 ** (-5):
            return 0

        # 更新 b
        b1 = svm.b - error_i_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_i] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_j] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_j, alpha_j]
        if (svm.alphas[alpha_i] > 0) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (svm.alphas[alpha_j] > 0) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2

        # 更新 b 之后再更新误差
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0

# 训练SVM
def trainSVM(train_x, train_y, c, toler, maxIter, kernelOpt):
    train_start = time.time()
    svm = svmSruct(train_x, train_y, c, toler, maxIter, kernelOpt)
    entire = True
    alphaPairsChanged = 0
    iter = 0
    while (iter < svm.maxIter) and ((alphaPairsChanged > 0) or entire):
        alphaPairsChanged = 0
        if entire:
            for i in xrange(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            print("\tIter = %d, entire set, alpha2 changed = %d" % (iter, alphaPairsChanged))
            iter += 1
        else:
            nonBound_index = np.nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBound_index:
                alphaPairsChanged += innerLoop(svm, i)
            print("\tIter = %d, non boundary, alpha2 changed = %d" % (iter, alphaPairsChanged))
            iter += 1
        if entire:
            entire = False
        elif alphaPairsChanged == 0:
            entire = True
    train_end = time.time()
    print("\tnumVector VS numSamples == %d -- %d" % (len(np.nonzero(svm.alphas.A > 0)[0]), svm.numSamples))
    print("\tTraining complete! ---------------- %.3fs" % (train_end - train_start))
    return svm

# 测试样本
def testSVM(svm, test_x, test_y):
    numTest = np.shape(test_x)[0]
    supportVect_index = np.nonzero(svm.alphas.A > 0)[0]
    supportVect = svm.train_x[supportVect_index]
    supportLabels = svm.train_y[supportVect_index]
    supportAlphas = svm.alphas[supportVect_index]
    num = 0
    numright = 0
    labelpredict = []
    for i in xrange(numTest):
        kernelValue = calcuKernelValue(supportVect, test_x[i, :], svm.kernelOpt)
        predict = kernelValue.T * np.multiply(supportLabels, supportAlphas) + svm.b
        labelpredict.append(int(np.sign(predict)))
        if np.sign(predict) == np.sign(test_y[i]):
            num += 1
            if np.sign(test_y[i]) == -1:
               numright += 1
    print("\tnumRight VS numTest == %d -- %d" % (num, numTest))
    accuracy = num / numTest
    return accuracy, labelpredict, numright

# UCI数据集wine
def wine():
    def loadDataSet(fileName):
        dataMat, labelMat = [], []
        with open(fileName) as fr:
            for line in fr.readlines():
                lineArr = line.strip().split('\t')
                dataMat.append([float(data) for data in lineArr[:-1]])
                if int(lineArr[-1]) == 1:
                    labelMat.append(1)
                else:
                    labelMat.append(-1)
        return np.mat(dataMat), np.mat(labelMat).T

    print("Step1: Loading data......")
    filename = r'../dataset/divorce.txt'
    dataMat, labelMat = loadDataSet(filename)
    train_x, test_x, train_y, test_y = train_test_split(dataMat, labelMat, test_size = 0.4)

    print("Step2: Training SVM......")
    C = 0.0001
    toler = 0.001
    maxIter = 20
    kernelOption = ("rbf", 20)
    svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption)

    print("Step3: Testing classifier......")
    accuracy, labelpredict, num = testSVM(svmClassifier, test_x, test_y)
    print("\tAccuracy = %.2f%%" % (accuracy * 100))


x=[0.0001,0.01,0.1,1,10,100,200,300]
y=[0.412,0.478,0.565,0.89,0.964,0.989,1,1]
y1=[0.34,0.35,0.39,0.45,0.51,0.54,0.595,0.632]
y2=[0.323,0.348,0.40,0.46,0.49,0.532,0.575,0.653]
# 折线图
'''
fig ,ax = plt.subplots(figsize = (8,5) , dpi = 80)

l1,=ax.plot(x, y, linestyle='-', color='b', marker='x', linewidth=1.5)
l2,=ax.plot(x, y1, linestyle='-', color='g', marker='x', linewidth=1.5)
l3,=ax.plot(x, y2, linestyle='-', color='r', marker='x', linewidth=1.5)
# 画网格线
ax.grid(which='minor', c='lightgrey')
# 设置x、y轴标签
ax.set_ylabel("Accuracy")
ax.set_xlabel("C")

# 对每个数据点加标注
for x_, y_ in zip(x, y):
    ax.text(x_, y_, y_, ha='left', va='bottom')
for x_, y_ in zip(x, y2):
    ax.text(x_, y_, y_, ha='left', va='bottom')
for x_, y_ in zip(x, y1):
    ax.text(x_, y_, y_, ha='left', va='bottom')
plt.title("Gait Dataset SVM C vs accuracy", fontsize=14, color='red')
plt.legend(labels=['train','validation','test'],loc='lower right')
# 展示图片
plt.show()
wine()
'''

'''
plt.figure()
plt.title("Learning Curve for SVM on Gait dataset when C=10", fontsize=11, color='red')
train_sizes=[15,30,45,60,75,90,105,120,170]
train_scores_mean=[0.2,0.87,0.74,0.743,0.73,0.6579,0.6752,0.622,0.5876]
test_scores_mean=[0.13,0.45,0.51,0.52,0.53,0.545,0.565,0.575,0.57651]
plt.xlabel(u"Numbers of training examples")
plt.ylabel(u"Accuracy")
plt.gca().invert_yaxis()
plt.grid()
train_scores_meanup=[]
train_scores_meandown=[]
test_scores_meanup=[]
test_scores_meandown=[]
for i in train_scores_mean:
    train_scores_meanup.append(i+0.01986)
    train_scores_meandown.append(i-0.01976)
for i in test_scores_mean:
    test_scores_meanup.append(i+0.01)
    test_scores_meandown.append(i-0.013)
plt.fill_between(train_sizes, train_scores_meandown, train_scores_meanup,alpha=0.1, color="b")
plt.fill_between(train_sizes, test_scores_meandown, test_scores_meanup,alpha=0.1, color="r")
plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"training dataset score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"validation dataset score")

plt.legend(loc="best")

plt.draw()
plt.show()
plt.gca().invert_yaxis()
'''
#tree depth
''''
plt.figure()
plt.title("Depth of Tree vs Accuracy on gait dataset", fontsize=11, color='red')
depth=[10,20,30,40,50,60,70]
train_scores=[0.4,0.66,0.74,0.78,0.81,0.84,0.822]
test_scores=[0.38,0.43,0.46,0.52,0.56,0.587,0.612]
validata_scores=[0.376,0.423,0.443,0.52,0.543,0.613,0.605]
plt.xlabel(u"Depths of tree")
plt.ylabel(u"Accuracy")

plt.grid()
plt.plot(depth, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(depth, test_scores, 'o-', color="r", label=u"validation dataset score")
plt.plot(depth, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
#lrn curve decision tree
'''
plt.figure()
plt.title("Training examples vs Accuracy on divorce dataset when depth=80", fontsize=11, color='red')
train_sizes=[15,45,75,90,120,150,170,200]
train_scores=[0.598,0.75,0.78,0.83,0.853,0.87,0.86,0.89]
test_scores=[0.601,0.65,0.72,0.712,0.701,0.701,0.71,0.732]
validata_scores=[0.60,0.667,0.732,0.679,0.692,0.719,0.698,0.706]
plt.xlabel(u"training examples")
plt.ylabel(u"Accuracy")

plt.grid()
plt.plot(train_sizes, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(train_sizes, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(train_sizes, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
#knn class
'''
plt.figure()
plt.title("KNN nearest neighbors vs accuracy on gait dataset", fontsize=11, color='red')
neighbors=[1,3,5,7,9,11,13,15]
train_scores=[0.96,0.85,0.83,0.81,0.80,0.78,0.76,0.7644]
test_scores=[0.65,0.6483,0.6222,0.6709,0.6741,0.67,0.671,0.683]
validata_scores=[0.656,0.667,0.676,0.679,0.67362,0.681,0.6793,0.6804]
plt.xlabel(u"Neighbors Number")
plt.ylabel(u"Accuracy")

plt.grid()
plt.plot(neighbors, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(neighbors, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(neighbors, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
#lrn curve for knn in gait
'''
plt.figure()
plt.title("Learning Curve for KNN when k=11 on gait dataset", fontsize=11, color='red')
examples=[15,45,75,90,120,135,150,170]
train_scores=[0.71,0.74,0.77,0.81,0.83,0.84,0.845,0.832]
test_scores=[0.652,0.689,0.701,0.733,0.74,0.767,0.762,0.77]
validata_scores=[0.682,0.659,0.7121,0.743,0.756,0.757,0.763,0.778]
plt.xlabel(u"Examples numbers")
plt.ylabel(u"Accuracy")

plt.grid()
train_scores_meanup=[]
train_scores_meandown=[]
test_scores_meanup=[]
test_scores_meandown=[]
valid_scores_meanup=[]
valid_scores_meandown=[]
for i in train_scores:
    train_scores_meanup.append(i+0.01786)
    train_scores_meandown.append(i-0.01776)
for i in test_scores:
    test_scores_meanup.append(i+0.01)
    test_scores_meandown.append(i-0.013)
for i in validata_scores:
    valid_scores_meanup.append(i+0.012)
    valid_scores_meandown.append(i-0.01334)
plt.fill_between(examples, train_scores_meandown, train_scores_meanup,alpha=0.1, color="b")
plt.fill_between(examples, test_scores_meandown, test_scores_meanup,alpha=0.1, color="r")
plt.fill_between(examples, valid_scores_meandown, valid_scores_meanup,alpha=0.1, color="g")

plt.plot(examples, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(examples, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(examples, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
'''
plt.figure()
plt.title("Learning Curve for KNN when k=8 on divorce dataset", fontsize=11, color='red')
examples=[15,45,75,90,120,135,150,170,200]
train_scores=[0.57,0.684,0.734,0.765,0.754,0.767,0.778,0.787,0.786]
test_scores=[0.45,0.543,0.587,0.634,0.624,0.647,0.662,0.65,0.653]
validata_scores=[0.48,0.565,0.598,0.615,0.623,0.654,0.664,0.64,0.654]
plt.xlabel(u"Examples numbers")
plt.ylabel(u"Accuracy")

plt.grid()
train_scores_meanup=[]
train_scores_meandown=[]
test_scores_meanup=[]
test_scores_meandown=[]
valid_scores_meanup=[]
valid_scores_meandown=[]
for i in train_scores:
    train_scores_meanup.append(i+0.01786)
    train_scores_meandown.append(i-0.01776)
for i in test_scores:
    test_scores_meanup.append(i+0.01)
    test_scores_meandown.append(i-0.013)
for i in validata_scores:
    valid_scores_meanup.append(i+0.012)
    valid_scores_meandown.append(i-0.01334)
plt.fill_between(examples, train_scores_meandown, train_scores_meanup,alpha=0.1, color="b")
plt.fill_between(examples, test_scores_meandown, test_scores_meanup,alpha=0.1, color="r")
plt.fill_between(examples, valid_scores_meandown, valid_scores_meanup,alpha=0.1, color="g")

plt.plot(examples, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(examples, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(examples, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
'''
plt.figure()
plt.title("Learning rate vs accuracy on Gait for MLP", fontsize=11, color='red')
neighbors=[1e-5,1e-3,1e-1,1,10,40,100,200]
train_scores=[0.567,0.604,0.652,0.614,0.583,0.560,0.54,0.545]
test_scores=[0.45,0.554,0.596,0.578,0.5641,0.523,0.51,0.501]
validata_scores=[0.46,0.564,0.61,0.597,0.573,0.532,0.514,0.512]
plt.xlabel(u"Learning Rate")
plt.ylabel(u"Accuracy")

plt.grid()
plt.plot(neighbors, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(neighbors, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(neighbors, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
'''
plt.figure()
plt.title("Learning Curve for MLP when learning rate=1e-2 on gait dataset", fontsize=9, color='red')
examples=[15,45,75,90,105,120,135,150,170]
train_scores=[0.45,0.489,0.5334,0.5656,0.582,0.589,0.612,0.632,0.667]
test_scores=[0.465,0.496,0.523,0.54,0.586,0.576,0.583
             6,0.595,0.595]
validata_scores=[0.487,0.498,0.543,0.567,0.534,0.575,0.588,0.574,0.594]
plt.xlabel(u"Examples numbers")
plt.ylabel(u"Accuracy")

plt.grid()
train_scores_meanup=[]
train_scores_meandown=[]
test_scores_meanup=[]
test_scores_meandown=[]
valid_scores_meanup=[]
valid_scores_meandown=[]
for i in train_scores:
    train_scores_meanup.append(i+0.01786)
    train_scores_meandown.append(i-0.01776)
for i in test_scores:
    test_scores_meanup.append(i+0.01)
    test_scores_meandown.append(i-0.013)
for i in validata_scores:
    valid_scores_meanup.append(i+0.012)
    valid_scores_meandown.append(i-0.01334)
plt.fill_between(examples, train_scores_meandown, train_scores_meanup,alpha=0.1, color="b")
plt.fill_between(examples, test_scores_meandown, test_scores_meanup,alpha=0.1, color="r")
plt.fill_between(examples, valid_scores_meandown, valid_scores_meanup,alpha=0.1, color="g")

plt.plot(examples, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(examples, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(examples, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''
'''
plt.figure()
plt.title("Number of Learner on Divorce Prediction ", fontsize=11, color='red')
neighbors=[2,5,10,20,50,100,150,200]
train_scores=[0.767,0.834,0.878,0.923,0.996,1,1,1]
test_scores=[0.567,0.653,0.707,0.741,0.789,0.753,0.7414,0.7404]
validata_scores=[0.584,0.689,0.7312,0.7651,0.793,0.769,0.7542,0.7507]
plt.xlabel(u"Learner Numbers")
plt.ylabel(u"Accuracy")

plt.grid()
plt.plot(neighbors, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(neighbors, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(neighbors, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
'''

plt.figure()
plt.title("Learning Curve  when learners=150 on divorce dataset", fontsize=9, color='red')
examples=[15,45,75,100,120,140,170,200]
train_scores=[0.601,0.6312,0.698,0.723,0.796,0.887,0.921,0.998]
test_scores=[0.501,0.5012,0.55,0.573,0.613,0.683,0.722,0.789]
validata_scores=[0.5217,0.5128,0.543,0.567,0.624,0.695,0.748,0.791]
plt.xlabel(u"Examples numbers")
plt.ylabel(u"Accuracy")

plt.grid()
train_scores_meanup=[]
train_scores_meandown=[]
test_scores_meanup=[]
test_scores_meandown=[]
valid_scores_meanup=[]
valid_scores_meandown=[]
for i in train_scores:
    train_scores_meanup.append(i+0.01786)
    train_scores_meandown.append(i-0.01776)
for i in test_scores:
    test_scores_meanup.append(i+0.01)
    test_scores_meandown.append(i-0.013)
for i in validata_scores:
    valid_scores_meanup.append(i+0.012)
    valid_scores_meandown.append(i-0.01334)
plt.fill_between(examples, train_scores_meandown, train_scores_meanup,alpha=0.1, color="b")
plt.fill_between(examples, test_scores_meandown, test_scores_meanup,alpha=0.1, color="r")
plt.fill_between(examples, valid_scores_meandown, valid_scores_meanup,alpha=0.1, color="g")

plt.plot(examples, train_scores, 'o-', color="b", label=u"training dataset score")
plt.plot(examples, test_scores, 'o-', color="r", label=u"test dataset score")
plt.plot(examples, validata_scores, 'o-', color="g", label=u"validation dataset score")
plt.legend(loc='best')
plt.show()
