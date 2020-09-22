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

To run: python kmeans.py [Number of images for training] [Flag_of_reading]

#### Algorithm description:
----------------------
1. The code starts from main() function where, initially flag and n value is read
2. The function NormalizeData, readCentersFromInput, readCentersRandomFromData does the job as the name suggests
3. The function doClustering is the one where final labels and centers are calculated
4. The function Classification calculates the distances and distributes the images into different clusters
5. The function UpdateCenter updates the centers for the new clusters
6. The function checkIfConvergence checks if the clusters are converged and ends the infinite while loop.
