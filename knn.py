def unpickle(file):
    #taken from https://www.cs.toronto.edu/~kriz/cifar.html
    import pickle
    import numpy
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def extractData(path):
    data = unpickle(path)

    list_labels = data[b'labels']
    list_data = data[b'data']

    labels = []
    data_set = []
# we are extracting first 1000 images
    for i in range(1000):
        labels.append(list_labels[i])
        data_set.append(list_data[i])
    return labels, data_set

import numpy as np

def ConvertRGB_grayscale(dataTrain, dataTest):
    #Y = 0.299R + 0.587G + 0.114B
    dataTrainGrayScale = []
    dataTestGrayScale = []

    for i in dataTrain:
        dataTrainGrayScale.append((i[0:1024:1] * 0.299) + (i[1024: 2048: 1] * 0.587) + (i[2048: 3072: 1] * 0.114))
    for i in dataTest:
        dataTestGrayScale.append((i[0:1024:1] * 0.299) + (i[1024: 2048: 1] * 0.587) + (i[2048: 3072: 1] * 0.114))
    np.array(dataTestGrayScale)
    np.array(dataTrainGrayScale)

    return dataTrainGrayScale, dataTestGrayScale

def doPreprocessing(GrayScaleTrainData, GrayScaleTestData, D):
    # taken from https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    from sklearn.decomposition import PCA
    pca = PCA(n_components=D, svd_solver='full')
    # pca.fit(GrayScaleTrainData)
    # pca.fit(GrayScaleTestData)
    PCAtestData = pca.fit_transform(GrayScaleTestData)
    PCAtrainData = pca.fit_transform(GrayScaleTrainData)
    return PCAtrainData, PCAtestData

def getDistance(p1, p2):
    #manhattan distance between two points p1 and p2
    distance = 0
    for i in range(len(p1)):
        distance += abs(p1[i] - p2[i])
    return distance


# taken from https://stackoverflow.com/questions/7971618/python-return-first-n-keyvalue-pairs-from-dict
from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def voting(KNN, Labels):
    sum = 0
    #dict with key as label and values as weight
    dict = {}
    for j in range(10):
        dict[j] = 0
    # K_Nearest_Neighbors is a list of tuples of index and distance
    for i in KNN:
        dict[Labels[i[0]]] += i[1]
    key_max = max(dict.keys(), key=(lambda k: dict[k]))
    return key_max

def doKNN(DataTrain, DataTest, TrainLabels, TestLabels, K):
    # in results we'll store predicted and ground truth labels
    results = []
    for testDataIndex, sample in enumerate(DataTest):
        # intializing a dict to store indexes and distances as key and value pairs
        #taken from https://www.geeksforgeeks.org/python-dictionary/
        Neighbor_distances_indices = {}
        for index, dataPoint in enumerate(DataTrain):
            distance = getDistance(sample, dataPoint)
            Neighbor_distances_indices[index] = distance
        # now sort the dict in increasing order of distances
        sortedDict = {k: v for k, v in sorted(Neighbor_distances_indices.items(), key=lambda item: item[1])}
        K_Nearest_Neighbors = take(K, sortedDict.items())
        #K_Nearest_Neighbors is now a list of tuples of index and distance
        #Now perform voting
        predictedLabel = voting(K_Nearest_Neighbors, TrainLabels)
        tup = (predictedLabel, TestLabels[testDataIndex])
        results.append(tup)

    return results

def KNNsklearn(dataTrain, dataTest, labelsTrain, labelsTest, K):
    # taken from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=K, weights='distance', p=1, metric='manhattan')
    neigh.fit(dataTrain, labelsTrain)
    results = []
    for index, sample in enumerate(dataTest):
        Predictedval = neigh.predict([sample])
        tup = (Predictedval[0], labelsTest[index])
        results.append(tup)
    return results

def doWrite(list, fileName):
    opfile = open(fileName, "w")
    for i in list:
        opfile.write(str(i[0]) + " " + str(i[1]) + "\n")
    opfile.close()

import matplotlib.pyplot as plt
def main():
    import sys
    K = int(sys.argv[1])
    D = int(sys.argv[2])
    N = int(sys.argv[3])
    PATH_TO_DATA = sys.argv[4]
    labels, data = extractData(PATH_TO_DATA)

    # dividing data into N train data and 1000-N test data
    dataTrain = data[0: N:1]
    dataTest = data[N: : 1]
    labelsTrain = labels[0: N : 1]
    labelsTest = labels[N : : 1]

    GrayScaleTrainData, GrayScaleTestData = ConvertRGB_grayscale(np.array(dataTrain), np.array(dataTest))
    PCAtrainData, PCAtestData = doPreprocessing(GrayScaleTrainData, GrayScaleTestData, D)

    #perform KNN
    results = doKNN(PCAtrainData, PCAtestData,labelsTrain, labelsTest, K)
    doWrite(results, "knn_results.txt")

    resultSklearn = KNNsklearn(dataTrain, dataTest, labelsTrain, labelsTest, K)
    doWrite(resultSklearn, "knn_results_sklearn.txt")


if __name__ == '__main__':
   main()


