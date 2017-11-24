#####################################################################################
# Bayesian classifer : Bayesian Classification of Two Types of Normal Distribution
# Author : luciaLiu
# Date   : 2017-11-12
# From   : Book page29
######################################################################################
import numpy as np


def cal_mean(sample):
    return np.mean(sample, axis=0)


def cal_conv(sample):
    return np.cov(sample)


def bayesian_classifer(sample1,sample2,p_w1,p_w2):
    m1 = cal_mean(sample1)
    m2 = cal_mean(sample2)
    conv1 = cal_conv(sample1)
    conv2 = cal_conv(sample2)
    dicision = np.log10(p_w1)-np.log10(p_w2)+(m1-m2).T

sample1 = {[],[],[],[]}
sample2 = {[],[],[],[]}
bayesian_classifer(sample1,sample2,0.5,0.5)