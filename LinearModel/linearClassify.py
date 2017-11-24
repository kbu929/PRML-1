#  Machine Learning class homework_1
# logistic Regression and 10 fold cross validation and Leave-One-Out
#
import numpy as np
import os


class Classifer(object):
    # 初始化分类器
    def __init__(self, attr_num, learn_rate=0.05):
        self.__attr_num___ = attr_num  # 属性个数
        self.__learn_rate__ = learn_rate  # 学习率
        self.__weight__ = np.zeros(shape=[attr_num + 1],
                                   dtype=np.float32)  # number of weights is var_num +1(beacause of b)

    def fit(self, value, label):
        value = np.append(value, [1.0])  # b
        linear_result = np.dot(value, self.__weight__)  # calculate the w.T *x
        sigmoid_result = 1.0 / (np.exp(-linear_result) + 1.0)  # calculte the value after sigmoid function
        for idx in range(self.__attr_num___ + 1):  # gradint descent
            update_val = (sigmoid_result - label) * value[idx]  # calculte update value
            self.__weight__[idx] -= self.__learn_rate__ * update_val  # update weights

    def classify(self, value):  # for classifying
        value = np.append(value, 1.0)  # add b
        linear_result = np.dot(value, self.__weight__)  # calculte the value after linear function
        sigmoid_result = 1.0 / (np.exp(-linear_result) + 1.0)  # logistic regession
        if sigmoid_result > 0.5:
            return 1
        else:
            return 0

    def save_weight(self, file_name='weight.npy'):  # save the weights value
        np.save(file_name, self.__weight__)

    def load_weight(self, file_name='weight.npy'):  # load weights
        if os.path.exists(file_name):
            self.__weight__ = np.load(file_name)
        else:
            raise RuntimeError('Can not find the file!')
