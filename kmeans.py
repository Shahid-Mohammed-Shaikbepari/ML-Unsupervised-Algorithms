#taken from https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
def read(num_images):
    import gzip
    f = gzip.open('train-images-idx3-ubyte.gz','r')
    image_size = 28
    import numpy as np
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = data.reshape(num_images, image_size, image_size, 1)

    #re shaped the images into an 2D array with number of image_number as row and image_size*image_size as column
    data = data.reshape(num_images, image_size * image_size)
    return data


def normalizeData(Data):
    #taken from https://stackoverflow.com/questions/52754284/how-to-create-array-with-many-normalized-float-values-in-python
    _range = range(0, 256) # range from [0, 255]
    NormalizedData = []
    for data in Data:
        _n_range = [(x - min(_range)) / (max(_range) - min(_range)) for x in data]
        NormalizedData.append(_n_range)
    return np.array(NormalizedData)


def main():
    #taken from https://www.pythonforbeginners.com/system/python-sys-argv
    import sys
    numImages = sys.argv[1]
    Flag_of_Reading = sys.argv[2]
    Data = read(int(numImages))
    NormalizedData = normalizeData(Data)
    if Flag_of_Reading == 1:
        Centers = readCentersFromInput('input.txt')
    else:
        Centers = readCentersRandomFromData(NormalizedData)
    labels, Centers = doClustering(NormalizedData, Centers)
    opfile = open("results.txt", "w")
    for label in labels:
        opfile.write(str(label+1) + "\n")
    opfile.close()

def doClustering(NormalizedData, Centers):
    #taken from http://madhugnadig.com/articles/machine-learning/2017/03/04/implementing-k-means-clustering-from-scratch-in-python.html
    Clusters = {} # declaration as a set
    for i in range(len(Centers)):
        Clusters[i] = [] #each cluster has list in it
    #test: check if Centers, Normalized_data is correct?

    while(1):
        # to classify the clusters with initial centers and after that with updated clusters
        Clusters = Classification(NormalizedData,Centers,Clusters)
        # store previous centers
        prevCenters = Centers.copy()
        Centers = UpdateCenters(Clusters, Centers)
        if checkIfConvergence(prevCenters, Centers):
            break

    labels = []
    #storing clusters indexes for each image
    for image in NormalizedData:
        distances = GetDistances(image, Centers)
        # get the index of centroid which has the minimum distance
        Index = distances.index(min(distances))
        labels.append(Index)
    return labels, Centers


def checkIfConvergence(prevCenters, Centers):
    isConverged = False
    tolerance = 0.00035
    for i in range(len(prevCenters)):
        sum = 0
        for j in range(len(Centers[0])): #distance between two points
            sum += (prevCenters[i][j] - Centers[i][j])**2
        dist = math.sqrt(sum)
        print(dist)
        if i == len(prevCenters)-1 and dist <= tolerance:
            isConverged = True
        elif dist <= tolerance: # if distance between them is not zero, centers not converged continue updating centers
            continue
        else:
            break

    return isConverged




def Classification(NormalizedData, Centers, Clusters):
    #measure distances from each point to the centers
    for image in NormalizedData:
        distances = GetDistances(image, Centers)
        #get the index of centroid which has the minimum distance
        Index = distances.index(min(distances))

        #add the image to the respective cluster
        Clusters[Index].append(image)
    return Clusters

import math

def GetDistances(image, Centers):
    distances = []
    for center in Centers:
        sum = 0
        for i in range(len(center)):
            sum += (image[i] - center[i])**2
        distance = math.sqrt(sum)
        distances.append(distance)
    return distances


def UpdateCenters(Clusters, Centers):
#taken from http://madhugnadig.com/articles/machine-learning/2017/03/04/implementing-k-means-clustering-from-scratch-in-python.html
    for i in range (len(Clusters)):
        Centers[i] = np.average(Clusters[i], axis=0)
    return Centers


from decimal import *
import numpy as np
import random
def readCentersFromInput(fileName):
    #taken from https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/
    file1 = open(fileName, 'r')
    Lines = file1.readlines()
    centers = []
    for line in Lines:
        #taken from https://stackoverflow.com/questions/19334374/python-converting-a-string-of-numbers-into-a-list-of-int
        center = [[float(s)] for s in line.split(',')]
        centers.append(center)
    file1.close()
    return np.array(centers)

def readCentersRandomFromData(Data):
    centers = []
    for i in range(7):
        centers.append(Data[random.randrange(1, len(Data)+1)])
    return np.array(centers)

if __name__ == '__main__':
     main()




