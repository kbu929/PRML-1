######################################
# k-Nearest Neighhbor Algorithm
# Author:luciaLiu
# Data:2017-11-15
# From: Pattern Recognition and Machine Learning book chapter 9
# from the matlab code of the mixGaussianEM.py
######################################
import numpy as np
import random


# creat the data from the "standard normal"distribution
#
# def create_data(mean, num_sampels, dims):
#     return np.random.rand(num_sampels, dims) + mean


# Expectation-Maximization interation implementation of Gaussian Mixtrure Model
# k_centroid可以是一个整数代表有几个类别，也可以是k个质心的二维列向量
# 如果是一个整数，那我就要产生随机的初始的k个centroid
#  - dataset:N-by-D data matrix.
#  - K_CENTROIDS: either K indicating the number of
#       components or a K-by-D matrix indicating the
#       choosing of the initial K centroids.
# def init_centroid(dataset, k_centroid):
#     num_samples, dims = dataset.shape
#     K = k_centroid
#     if isinstance(K, int):  # if k_centroid is a number means the number of the component
#         centroids = np.zeros((K, dims))
#         for i in range(K):
#             index = int(random.uniform(0, num_samples))
#             centroids[i, :] = dataset[index, :]  # return the number of the cluster and centroid of each cluster
#     else:  # if k_centroid is a initial K centroid
#         K = np.array(k_centroid).shape[0]
#         centroids = np.array(k_centroid)
#     return K, centroids
#
#
# # calculate the value of likelihood function
#
# # initial the parameters
# #  - MODEL: a structure containing the parameters for a GMM:
# #       MODEL.centroid_mean: a K-by-D matrix.
# #       MODEL.centroid_sigma: a D-by-D-by-K matrix.
# #       MODEL.Pi_k: a 1-by-K vector.
# # computer the pdf of the multi-var gaussian
# # 按照模式识别课本26页的公式
# def gaussian(x, miu, sigma):
#     norm_factor = (2 * np.pi) ** len(x)
#     norm_factor *= np.linalg.det(sigma)  # np.linalg.det计算矩阵的行列式
#     norm_factor = 1.0 / np.sqrt(norm_factor)
#     x_mu = np.matrix(x - miu)
#     rs = norm_factor * np.exp(-0.5 * x_mu.T * np.linalg.inv(sigma) * x_mu)
#     return rs
#
#
# # calculate the value of likelihood function
# def likelihood(x, k, miu, sigma, pi):
#     likelihood = 0.0
#     for k in range(k):
#         likelihood += pi[k] * gaussian(x, miu[k], sigma[k])
#     return likelihood
#
#
# def log_likelihood(data, k, miu, sigma, pi):
#     log_likelihood = 0.0
#     for n in range(len(data)):
#         log_likelihood += np.log(likelihood(data[n], k, miu, sigma, pi))
#     return log_likelihood
#
#
# def init_param(data, k):
#
#     centroid_mean = init_centroid(data,k)
#     pi_k = np.zeros(1, k)
#     centroid_sigma = np.zeros((centroid_mean.shape[0], centroid_mean.shape[0], k))
#     # evaluate the initial value of the log likelihood
#     llh_value = log_likelihood(data, k, centroid_mean, centroid_sigma, pi_k)
#     return centroid_mean, centroid_sigma, pi_k, llh_value
# ##gaussian mixture model
class mixGaussianEM(object):
    print('EM for Gaussian mixture:running....')
    tol = 1e-6
    maxiter = 500
    llh = []
    for i in range(maxiter):
        llh.append(float('-Inf'))
    R = initialization(X,init)
    for iter in range(2,maxiter):
