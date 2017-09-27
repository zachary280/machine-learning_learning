from math import log


def computerEntropy(dataSet):
    #computer information entropy
    #dataSet is set format [property1[,property2[,...],result]

    setlength = len(dataSet)
    labelDirc = {}
    for set in dataSet:
        currentLabel = set[-1]
        if currentLabel in labelDirc.keys():
            labelDirc[currentLabel] += 1
        else:
            labelDirc[currentLabel] = 1

    entropy = 0.0
    for label in labelDirc.keys():
        probability = float(labelDirc[label]) / setlength
        entropy -= probability * log(probability, 2)
    return entropy



def computeGini(dataSet):
    # compute information gini

    setlength = len(dataSet)
    labelDirc = {}
    for set in dataSet:
        currentLabel = set[-1]
        if currentLabel in labelDirc.keys():
            labelDirc[currentLabel] += 1
        else:
            labelDirc[currentLabel] = 1
    gini = 0.0
    for label in labelDirc.keys():
        probability = float(labelDirc[label]) / setlength
        gini += probability * probability
    gini = 1 - gini
    return gini


def computeMinGiniIndex(dataSet):
    # CART decision making tree arithmetic

    setLength = len(dataSet)
    minGini = 100000
    setIndex = 0
    gini = {}
    for index in range(len(dataSet[0]) - 1):
        labelDirc = {}
        setDirc = {}
        for set in dataSet:
            currentLabel = set[index]
            if currentLabel in labelDirc:
                labelDirc[currentLabel] += 1
                setDirc[currentLabel].append(set)
            else:
                labelDirc[currentLabel] = 1
                setDirc[currentLabel] = []
                setDirc[currentLabel].append(set)
        currentGini = 0.0
        currentGiniIndex = 0.0
        for label in labelDirc.keys():
            currentGini = computeGini(setDirc[label])
            probability = float(labelDirc[label]) / setLength
            currentGiniIndex += probability * currentGini

        gini[index] = currentGiniIndex
        if currentGiniIndex < minGini:
            minGini = currentGiniIndex
            setIndex = index
    return minGini, setIndex, gini


def computerMaxGain(dataSet):
    # computer information Gain
    # dataSet is set format [property1[,property2[,...],result]
    # ID3 arithmetic

    # compute original data entropy
    oldEntropy = computerEntropy(dataSet)
    setLength = len(dataSet)
    # find max information Gain
    maxGain = 0
    # get property when get max information Gain
    setIndex = 0
    # get information gain for every property
    gain = {}
    # compute intrinstic value
    intrinsticValue = {}

    # consider every property
    for index in range(len(dataSet[0])-1):
        labelDirc = {}
        setDirc = {}

        # get set and the data in the set has same property
        for set in dataSet:
            currentLabel = set[index]
            if currentLabel not in labelDirc:
                labelDirc[currentLabel] = 0
                setDirc[currentLabel] = []
            labelDirc[currentLabel] += 1
            setDirc[currentLabel].append(set)
        # compute information gain
        currentEntropy = 0.0
        currentGain = 0.0
        iv = 0.0
        for label in labelDirc.keys():
            currentEntropy = computerEntropy(setDirc[label])
            probability = float(labelDirc[label]) / setLength
            currentGain += probability * currentEntropy
            iv -= probability*(log(probability,2))
        currentGain = oldEntropy - currentGain
        gain[index] = currentGain
        intrinsticValue[index] = iv
        # get max gain
        if currentGain >= maxGain:
            maxGain = currentGain
            setIndex = index
    return maxGain, setIndex, gain,intrinsticValue


def computeMaxGainContinuous(dataSet):
    # 计算连续值的信息增益
    oldEntropy = computerEntropy(dataSet)
    setLength = len(dataSet)
    # find max information Gain
    maxGain = 0
    # get property when get max information Gain
    setIndex = 0
    # get information gain for every property
    gain = {}
    # compute intrinstic value
    intrinsticValue = {}

    return

def computeGainRation(dataSet):
    # c4.5 arithmetic
    # compute information gain ration

    maxGain, setIndex, gain, intrinsticValue = computerMaxGain(dataSet)
    maxGainRation = 0
    setLable = 0
    gain_ration = {}
    # sort gain by dictionary items
    dirSorted1=sorted(gain.items(),key=lambda item:item[1])
    gainSorted = {}
    match = {}
    i = 0
    for item in dirSorted1:
        gainSorted[i]=item[1]
        match[i] = item[0]
        i += 1
    midIterator = len(gainSorted)//2
    for label in gainSorted.keys():
        if label+1 >= midIterator:
            # IV 会出现0的情况不能做被除数
            if(intrinsticValue[match[label]] == 0):
                result = 0
            else:
                result = gainSorted[label] / intrinsticValue[match[label]]
            gain_ration[label] = result
            if result >= maxGainRation:
                maxGainRation = result
                setLable = label
    return maxGainRation, gain_ration, match[setLable]


# 使用此算法创建决策树，将会产生两个决策树，这两个决策树的结构一样，但是存在左右孩子对调的情况
# 使用此算法的信息增益算法和增益率算法产生的决策树一样，未找到问题所在
def treeGenerate(dataSet, labels):
    # generate decision making tree
    classList = [example[-1] for example in dataSet]
    # 判断样本是否属于同一个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 判断标签值是否为空或者样本的属性是否都相同
    if len(labels) == 0 or  sameLable(dataSet,labels):
        return majorityCnt(classList)
    # 可以进行替换，以使用不同的算法
    #  信息增益算法
    maxGain, setIndex, gain, intrinsticValue=computerMaxGain(dataSet)
    # 使用增益率算法
    #maxGainRation, gain_ration, setIndex = computeGainRation(dataSet)
    # 使用基尼指数
    # minGini, setIndex, gini = computeMinGiniIndex(dataSet)
    bestFeatLable = labels[setIndex]
    myTree = {bestFeatLable: {}}
    labelsTemp = labels[:]
    del (labelsTemp[setIndex])
    featValues = [example[setIndex] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, setIndex, value)
        if len(subDataSet) == 0:
            return majorityCnt(classList)
        else:
            myTree[bestFeatLable][value] = treeGenerate(subDataSet, labelsTemp)
    return myTree

# 暂时没法完成，使用信息增量获得的决策树的正确率为80%+，剪枝后正确率无法提高。
def treeGeneratePostPruning(dataSet, testSet, labels):
    # 使用后剪枝算法生成决策树
    # 1，生成普通决策树
    # 2，计算树的正确率
    # 3，将每一个叶节点删除，并以这个节点的父节点的所有样本最多的类别进行标记，计算正确率
    # 4，若正确率提高则重复步骤3，否则还原被删除的叶节点

    # 得到普通的决策树
    myTree = treeGenerate(dataSet, labels)
    print(myTree)
    # 计算树的正确率
    newAccuracy = accuracy(myTree, testSet, labels)
    return newAccuracy


def treeGeneratePrePruning(dataSet, testSet, labels, labelTemp, mytree={}, queue=[], maxAccuracy=0.0):
    # 本函数经过小范围的数据测试能够通过测试，但是对于大量的数据情况没有进行测试，运行结果未知

    # 使用预剪枝算法创建决策树,预剪枝会产生欠拟合的风险，能够减少过拟合的风险
    # 1 将节点定义为叶节点，将其类别标记为样本数最多的类
    # 2 使用测试数据集进行测试，得出正确率
    # 3 以此节点为根节点进行划分，得到节点，重复1,2步骤
    # 4 比较划分后的正确率，若提高则此节点进行划分，否则此节点终止划分

    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 判断标签值是否为空或者样本的属性是否都相同
    if len(labelTemp) == 0 or  sameLable(dataSet,labelTemp):
        return majorityCnt(classList)
    # 可以进行替换，以使用不同的算法
    # 信息增益算法
    maxGain, setIndex, gain, intrinsticValue=computerMaxGain(dataSet)
    # 使用增益率算法
    # maxGainRation, gain_ration, setIndex = computeGainRation(dataSet)
    # 使用基尼指数
    # minGini, setIndex, gini = computeMinGiniIndex(dataSet)
    bestFeatLable = labelTemp[setIndex]
    result = majorityCnt(classList)
    # 存在问题, 字典是可变对象，不能使用此方法操作
    # mytreeTemp = mytree
    # for iterator in range(len(queue)-1):
    #         mytreeTemp = mytreeTemp[queue[iterator]]
    # mytreeTemp[queue[-1]] = result

    # 创建子节点
    if len(queue) == 0:
        mytree = {bestFeatLable:{}}
        mytree[bestFeatLable] = result
        queue.append(bestFeatLable)
    else:
        oldValue = getOldValue(mytree,queue[:])
        buildTree(mytree,{},queue[:])
        queue.append(bestFeatLable)
        buildTree(mytree,result, queue[:])
    # 获取测试集进行测试的正确率
    newAccuracy = accuracy(mytree,testSet,labels)

    if newAccuracy > maxAccuracy:
        maxAccuracy = newAccuracy
        # 删除标签值
        del (labelTemp[setIndex])
        # 将相同属性的样本取出
        featValues = [example[setIndex] for example in dataSet]
        uniqueVals = set(featValues)
        # 创建子节点
        buildTree(mytree, {}, queue[:])
        queueTemp = queue [:]
        for value in uniqueVals:
            # 得到相同属性的样本
            subDataSet = splitDataSet(dataSet, setIndex, value)
            if len(subDataSet) == 0:
                pass
            else:
                queueTemp.append(value)
                classList = [example[-1] for example in subDataSet]
                # 创建子节点
                buildTree(mytree, majorityCnt(classList), queueTemp[:])
                queueTemp.pop()

        newAccuracy = accuracy(mytree, testSet, labels)
        if newAccuracy > maxAccuracy:
            maxAccuracy = newAccuracy
            for value in uniqueVals:
                # 得到相同属性的样本
                subDataSet = splitDataSet(dataSet, setIndex, value)
                if len(subDataSet) == 0:
                    pass
                else:
                    queue.append(value)
                    treeGeneratePrePruning(subDataSet, testSet, labels, labelTemp, mytree, queue,maxAccuracy)
                    del queue[-1]
    else:
        # 将树还原，删除最后添加的子节点，同时还原原先的子节点(难点)
        delTree(mytree, queue[:])
        del queue[-1]
        buildTree(mytree,oldValue ,queue[:])
    return mytree



def  treeGenerateContinuousValue():
    # 对于连续值的处理方法：采用二分法进行处理

    return

def getOldValue(tree, queue):
    if len(queue) == 1:
        oldValue = tree[queue[0]]
        return oldValue
    else:
        key = queue[0]
        del queue[0]
        oldValue = getOldValue(tree[key], queue)
    return oldValue

def delTree(tree,queue):
    if len(queue) == 1:
        tree.pop(queue[0])
    else:
        key = queue[0]
        del queue[0]
        delTree(tree[key],queue)
    return

def buildTree(tree, value, queue):
    if len(queue) == 1:
        tree[queue[0]] = value
    else:
        key = queue[0]
        del queue[0]
        buildTree(tree[key],value,queue)
    return


def accuracy(mytree, testSet, labels):
    setLength = len(testSet)
    current = 0

    for set in testSet:
        result = testData(set[:-1], labels, mytree)
        if result == set[-1]:
            current += 1
    return  float(current)/setLength

import operator

def majorityCnt(classList):
    # sort
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet, axis, value):
    # Partitioning data sets beyond features
    # axis is features , value is return value beyond feature
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def sameLable(dataSet,labels):
    for label in range(len(labels)):
        classList = [example[label] for example in dataSet]
        if classList.count(classList[0]) != len(classList):
            return False
    return True



def testData(data, labels, mytree):
    # 测试数据，通过data的属性的不同值，根据决策数来决定最终的结果

    # 将两个list 转化为地点，即将数据的属性值与属性对应转化为字典
    dataDict = dict(zip(labels, data))
    firstKey = list(mytree.keys())[0]
    if isinstance(mytree[firstKey],dict):
        if isinstance(mytree[firstKey][dataDict[firstKey]], dict):
            result = testData(data, labels,mytree[firstKey][dataDict[firstKey]])
        else:
            result = mytree[firstKey][dataDict[firstKey]]
    else:
        result = mytree[firstKey]
    return result
