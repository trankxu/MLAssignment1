# -*- coding: utf-8 -*-
import Decision_Tree_ID3 as TreeID3
import Decision_Tree_Visual as TreeVisual
import pandas as pd




#计算决策树在验证集的精度

def valPrecision(thisTree,valdata):
    classTrue=list(valdata.iloc[:,-1])
    valNum=valdata.shape[0]
    classPred=[]
    crtNum=0 #初始化预测正确样例数
    for rowNum in range(valNum):
        classSple=TreeID3.classify(thisTree,watermelonVal.iloc[rowNum,:]) #预测该样例的分类
        classPred.append(classSple)
        if classTrue[rowNum] == classSple: #判断预分类测是否正确
            crtNum+=1
    print("The validate accuracy is: {}".format(crtNum/valNum))
    return crtNum/valNum #返回分类精度


#对已建立的决策树进行后剪枝
#递归调用通过设置剪枝代码位置实现自底向顶或自顶向底进行剪枝
def createPostpruningTree(inputTree,dfdata,valdata):
    firstStr=list(inputTree.keys())[0] #获取第一个属性值
    secondDict=inputTree[firstStr]
    typedfdata=TreeID3.typeMajority(dfdata) #多数表决发确定剩余训练集的类别
    pruningTree={firstStr:{}} #初始化后剪枝决策树
    contrastTree={firstStr:{}} #对该属性建立不划分决策树
    for key in secondDict.keys():
        contrastTree[firstStr][key]=typedfdata #不划分决策树即每个属性取值样例的类别都为多数表决结果
        #以递归调用方式完善决策树
        if type(secondDict[key]).__name__=='dict':
            pruningTree[firstStr][key]=createPostpruningTree(secondDict[key],TreeID3.splitDataset(dfdata,firstStr,key),TreeID3.splitDataset(valdata,firstStr,key))
        else:
            pruningTree[firstStr][key]=secondDict[key]
    #针对该属性，计算剪枝后与不剪枝决策树在验证集的预测精度
    precisionContrast=valPrecision(contrastTree,valdata)
    precisionPruning=valPrecision(pruningTree,valdata)
    #将两种决策树进行比较，如果剪枝后能提高精度，则选择对该属性剪枝
    #剪枝操作放在递归调用之后，实现自底向顶的剪枝
    if precisionContrast>precisionPruning:
        #print(firstStr)
        #print(typedfdata)
        return typedfdata
    else:
        return pruningTree




if __name__== '__main__':
    watermelonTra = pd.read_csv('../gait_tra.csv', encoding='gbk')
    # 读取验证集
    watermelonVal = pd.read_csv('../gait_valid.csv', encoding='gbk')

    # 基于训练集创建未剪枝决策树
    treeOriginal = TreeID3.creatDecisionTree(watermelonTra)
    # 可视化未剪枝决策树
    TreeVisual.createTree(treeOriginal, '未剪枝决策树')
    #基于未剪枝决策树、训练集与验证集创建后剪枝决策树
    treePostpruning=createPostpruningTree(treeOriginal,watermelonTra,watermelonVal)
    #后剪枝决策树可视化
    TreeVisual.createTree(treePostpruning,'PostPruning Tree')


'''
# 基于训练集与验证集创建预剪枝决策树
treePrepruning = createPrepruningTree(watermelonTra, watermelonVal)
# 预剪枝决策树可视化
TreeVisual.createTree(treePrepruning, '预剪枝决策树')
# 创建西瓜数据集2.0的决策树
watermelonDecisionTree = creatDecisionTree(watermelon)
print(watermelonDecisionTree)
# 决策树可视化
Decision_Tree_Visual.createTree(watermelonDecisionTree, "ID3Decision_Tree Gait Dataset")
'''




