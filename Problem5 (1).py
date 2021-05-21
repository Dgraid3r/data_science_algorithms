import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial


#from scipy import spatial
def euclideanDistance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1-point2)

def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    d = len(X[0])
    n = len(X)
    centroids = []
    #np.random.random_sample((k, d))*d - 1 would work for regular k++
    #randomly select the first center
    centroids.append(X[np.random.randint(n)])
    
    for centers in range(1,k):
        distance = []
        for i in range(n):
            #we will compute the distance from point i to all centroids
            #we will assign point i to variable cur
            #we will use a basic method to find min
            cur = X[i]
            minDist = np.Inf
            for j in range(len(centroids)):
                temp = euclideanDistance(cur, centroids[j])
                minDist = min(minDist, temp)
            distance.append(minDist)
            
        #now we will pick our next center
        #pick a data point x to be the next ci with probability proportional to its distance
        #from current centers(the shortest distance from a point to a center that
        #we already have chosen)
        distance = np.array(distance)
        probability = distance/distance.sum()
        #overall = probability.cumsum()
        #now we need to randomly sample based on the probability we just calculated
        tempr = np.array(list(range(0,150)))
        new_centroid = np.random.choice(tempr, p = probability)
        centroids.append(X[new_centroid])
    return centroids



def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    centroids = k_init(X,k)
    for i in range(max_iter):
        #compute pairwise distance between each datapoint and centroid
        pair_dist = sp.spatial.distance.cdist(X, centroids)
        #assign datapoints to their nearest centroids
        Y = np.argmin(pair_dist, axis=1)
        temp = np.array(X)
        #calculate the mean of each centroid's data points and 
        #make that the new mean
        for j in range(k):
            centroids[j] = np.mean(temp[Y == j], axis=0)
    
    return centroids

    



def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    
    #for each data point compute the distance to the centers/centroids and assign
    #it to the closest one
    pair_dist = sp.spatial.distance.cdist(np.array(X), C)
    #assign datapoints to their nearest centroids
    Y = np.argmin(pair_dist, axis=1)
    #now just need to reformat the Y to datamap
    #creating an empty binary matrix of 0s
    matrix = np.zeros((150, len(Y)))
    #assigning 1 to all the places where Y has assigned the point to
    matrix[np.arange(Y.size),Y] = 1
        
    return matrix


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    #iterate through all of the data points and the final cluster centers and compute
    #the sum of all distances between each point and their assigned clusters
    
    #calculating distance between centroids and points + assigning data to clusters
    pair_dist = sp.spatial.distance.cdist(X, C)
    y = np.argmin(pair_dist, axis=1)
    
    #creating a dictionary to store the points so it makes summing easier/data neater
    dist = {x:[] for x in range(len(C))}
    totalDist = 0
    for i in range(len(C)):
        for j in range(len(X)):
            #if the current datapoint corresponds with the current cluster
            if(y[j] == i):
                cur = pair_dist[j][i]
                dist[i].append(cur)
    for k in range(len(dist)):
        totalDist += totalDist + sum(dist[k])
                
    return totalDist


#data preprocssing

cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv('iris.csv', names = cols)
iris['x1'] = iris['sepal_length']/iris['sepal_width']
iris['x2'] = iris['petal_length']/iris['petal_width']
xone = iris['x1']
xtwo = iris['x2']
X = pd.DataFrame(iris[['x1', 'x2']])
X = X.values.tolist()
#k = len(X)
k = 1
max_iter = 50
#this is just testing/loading in data bit I guess

#objective changing stuff
one = []
two = []
#running for k=1 to 5 to find optimal k
k=1
a = k_init(X,k)
b = k_means_pp(X, k, max_iter)
c = assign_data2clusters(X, b)
d = compute_objective(X,b)
one.append(k)
two.append(d)
k=2
a = k_init(X,k)
b = k_means_pp(X, k, max_iter)
c = assign_data2clusters(X, b)
d = compute_objective(X,b)
one.append(k)
two.append(d)
k=3
a = k_init(X,k)
b = k_means_pp(X, k, max_iter)
c = assign_data2clusters(X, b)
d = compute_objective(X,b)
one.append(k)
two.append(d)
k=4
a = k_init(X,k)
b = k_means_pp(X, k, max_iter)
c = assign_data2clusters(X, b)
d = compute_objective(X,b)
one.append(k)
two.append(d)
k=5
a = k_init(X,k)
b = k_means_pp(X, k, max_iter)
c = assign_data2clusters(X, b)
d = compute_objective(X,b)
one.append(k)
two.append(d)

#graphing to find optimal k
plt.figure(0)
plt.plot(one, two)

#highest "elbow" is at 4

#plot for optimal value of k, k=4
k = 4
a = k_init(X,k)
b = k_means_pp(X, k, max_iter)
bone = []
btwo = []
for i in range(len(b)):
    bone.append(b[i][0])
    btwo.append(b[i][1])
plt.figure(1)
plt.scatter(xone,xtwo, marker = '.', 
               color = 'red', label = 'data points')
plt.scatter(bone,btwo, 
               color = 'blue', label = 'centroids')
