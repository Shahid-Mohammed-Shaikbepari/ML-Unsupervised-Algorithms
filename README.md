# ML-Unsupervised-Algorithms
Machine Learning Unsupervised algorithms viz. K Nearest Neighbors and K Means clustering
#### Shahid Mohammed Shaikbepari
###### shaikbep@usc.edu

## K Means:
K-Means clustering aims to partition N observations into K clusters in which each observation belongs to the cluster with the nearest distance to its center. 
The K-Means has a loose relationship to the k-nearest neighbors classifier and the main difference between them is that there is no label used in K-Means clustering. 
#### Dataset:
The MNIST dataset has digit images (28x28 each) with 10 classes (0~9). There are in total 60000 images for training and 10000 images for testing. 

1. Download the MNIST training set image (“train-images-idx3-ubyte.gz”) from the link below: http://yann.lecun.com/exdb/mnist/index.html
 
2. Look into the description at the end of the web page (http://yann.lecun.com/exdb/mnist/index.html) to figure out how to extract the data from the downloaded files.

###### To run: python kmeans.py [Number of images for training] [Flag_of_reading]

#### Algorithm description:
----------------------
1. The code starts from main() function where, initially flag and n value is read
2. The function NormalizeData, readCentersFromInput, readCentersRandomFromData does the job as the name suggests
3. The function doClustering is the one where final labels and centers are calculated
4. The function Classification calculates the distances and distributes the images into different clusters
5. The function UpdateCenter updates the centers for the new clusters
6. The function checkIfConvergence checks if the clusters are converged and ends the infinite while loop.

## K-Nearest Neighbors (KNN) Classifier 
K-nearest neighbors algorithm (KNN) is a nonparametric method used for classification. A query object is classified by a majority vote of the K closest training examples (i.e. its neighbors) in the feature space. In this problem, the objective is to implement a KNN classifier to perform image classification given the image features obtained by PCA

#### Dataset:
The CIFAR-10 dataset has labeled tiny images (32x32 each) with 10 classes. There are in total 50000 images for training and 10000 images for testing. It is a widely-used dataset for benchmarking image classification models. 
•	Download the python version of CIFAR-10 dataset from the link below: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
•	Look into the description on the web page(https://www.cs.toronto.edu/~kriz/cifar.html)  to figure out how to extract the data from the downloaded pickle files.

#### Pre processing: 
Convert the RGB images to grayscale with the following formula: Y = 0.299R + 0.587G + 0.114B

###### To run: python knn.py K D N PATH_TO_DATA

#### Algorithm description:
----------------------
1. Read the K D N Path values from console
2. Divide the data into train and test data
3. Convert the images into grayscale with given formula in the description
4. Perform preprocessing on the data from the sklearn PCA module
5. Finally in the function doKNN, calculate the distance of the test point from each training point and note down first N values with least distance
6. Perform voting based on the weights = 1/distance and assign the class with max votes
7. Repeat the experiment with sklearn library’s KNeighbors Classifier
8. The outputs are reported in knn_results.txt and knn_results_sklearn.txt files


