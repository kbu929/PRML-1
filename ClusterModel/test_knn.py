#################################################
# kmeans: k-means cluster
# Author : luciaLiu
# Date   : 2017-11-12
# TestDataset:西瓜分类
#################################################
import numpy as np
import KNN
import matplotlib.pyplot as plt

# load dataset
print('Step1:load data...')
dataset = []
file_in = open('./data_xigua3.txt')
for line in file_in.readlines():
    # line.strip(rm) delete the 'rm' at the begining and ending
    # line.split example:s = "a#b#c",ls = line.split("#") ls = ['a','b','c']
    line_arr = line.split(' ')
    dataset.append([float(line_arr[0]), float(line_arr[4].strip())])
print('Step2"clustering')
dataset = np.mat(dataset)
k = 3
centroids, cluster_assment = KNN.k_means(dataset, k)
print('Step3:show the result')
KNN.show_cluster(dataset, k, centroids, cluster_assment)
