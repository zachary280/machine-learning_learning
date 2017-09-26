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
    # maxGain, setIndex, gain, intrinsticValue=computerMaxGain(dataSet)
    # 使用增益率算法
    # maxGainRation, gain_ration, setIndex = computeGainRation(dataSet)
    # 使用基尼指数
    minGini, setIndex, gini = computeMinGiniIndex(dataSet)
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
            subLabels = labelsTemp[:]
            myTree[bestFeatLable][value] = treeGenerate(subDataSet, labelsTemp)
    return myTree

def treeGeneratePrePunning(dataSet, testSet, labels):
    # 使用预剪枝算法创建决策树
    # 1 将节点定义为叶节点，将其类别标记为样本数最多的类
    # 2 使用测试数据集进行测试，得出正确率
    # 3 以此节点为根节点进行划分，得到节点，重复1,2步骤
    # 4 比较划分后的正确率，若提高则此节点进行划分，否则此节点终止划分

    classList = [example[-1] for example in dataSet]
    # 可以进行替换，以使用不同的算法
    # 信息增益算法
    maxGain, setIndex, gain, intrinsticValue=computerMaxGain(dataSet)
    # 使用增益率算法
    # maxGainRation, gain_ration, setIndex = computeGainRation(dataSet)
    # 使用基尼指数
    # minGini, setIndex, gini = computeMinGiniIndex(dataSet)
    bestFeatLable = labels[setIndex]
    myTree = {bestFeatLable: {}}
    result = majorityCnt(classList)
    myTree[bestFeatLable] = result
    # 获取测试集进行测试的正确率
    oldAccuracy = accuracy(myTree,testSet,labels)
    
    # 计算旧值
    return oldAccuracy

def accuracy(mytree, testSet, labels):
    setLength = len(testSet)
    current = 0

    for set in testSet:
        result = testData(set[:-2], labels, mytree)
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
