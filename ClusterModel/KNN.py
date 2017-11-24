######################################
# k-Nearest Neighhbor Algorithm
# Author:luciaLiu
# Data:2017-11-11
######################################
import numpy as np
import random
import matplotlib.pyplot as plt


# calculate Euclidean distance

def cal_Eudist(vec_x, vec_y):
    return np.sqrt(np.sum(np.power(np.abs(vec_x - vec_y), 2)))


# init centroid with random samples

def init_centroid(dataset, k):  # k is the center of the cluster
    num_samples, dims = dataset.shape  # number of dataset and the dimension
    centroid = np.zeros((k, dims))
    for i in range(k):
        index = int(random.uniform(0, num_samples))  # generate a random float between the min and max
        centroid[i, :] = dataset[index, :]
    return centroid


def k_means(dataset, k):
    num_samples, dims = dataset.shape
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    cluster_assment = np.mat(np.zeros((num_samples, 2)))
    cluster_changed = True
    # step1:init the centroid
    centroid = init_centroid(dataset, k)
    while cluster_changed:
        cluster_changed = False
        for i in range(num_samples):
            min_dist = 10000
            min_index = 0
            # find the closest centroid for each sample
            for j in range(k):
                dist = cal_Eudist(dataset[i, :], centroid[j, :])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            # update its cluster through update its centroid
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
                cluster_assment[i, :] = min_index, min_dist
        # update centroids
        for j in range(k):
            # find the sample whose index is j
            point_incluster = dataset[np.nonzero(cluster_assment[:, 0].A == j)[0]]
            # new centroid is mean value of the new cluster
            centroid[j, :] = np.mean(point_incluster, axis=0)
    print('cluster complete!')
    return centroid, cluster_assment


# show cluster only available with 2D data
def show_cluster(dataset, k, centroid, cluster_assment):
    num_samples, dims = dataset.shape
    if dims != 2:
        print('Sorry!Can not show the data because the dimension of it is not 2!')
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print('Sorry! your k is too large')
        return 1
    for i in range(num_samples):
        mark_index = int(cluster_assment[i, 0])
        plt.plot(dataset[i, 0], dataset[i, 1], mark[mark_index])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroid[i, 0], centroid[i, 1], mark[i], markersize=12)

    plt.show()
