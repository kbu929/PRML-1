######################################
# Patternn Classification Section5
# Author:luciaLiu
# Data:2017-11-21
######################################
import numpy as np
import xlrd
import matplotlib
import matplotlib.pyplot as plt
import random
import math

# load data
f = xlrd.open_workbook('./PatternClassification_5.xls')
sh = f.sheet_by_index(0)
x1 = sh.col_values(0)
x2 = sh.col_values(1)
label = sh.col_values(2)
# visulization
plt.figure(1)
idx_0 = [i for i, x in enumerate(label) if x == 0]
data = np.zeros((40, 3))
#增广矩阵
data[:, 0] = x1
data[:, 1] = x2
data[:, 2] = 1
p0 = plt.scatter(data[idx_0, 0], data[idx_0, 1], marker='*', color='m', label='0', s=30)
idx_1 = [i for i, x in enumerate(label) if x == 1]
p1 = plt.scatter(data[idx_1, 0], data[idx_1, 1], marker='*', color='g', label='1', s=30)
idx_2 = [i for i, x in enumerate(label) if x == 2]
p2 = plt.scatter(data[idx_2, 0], data[idx_2, 1], marker='o', color='r', label=2, s=25)
idx_3 = [i for i, x in enumerate(label) if x == 3]
p3 = plt.scatter(data[idx_3, 0], data[idx_3, 1], marker='x', color='b', label=3, s=15)
plt.legend(loc='upper right')
plt.show()
# basic gradient descent
# loss function is MSE(mininum square error)
w = [20,50,45] #initial value for weight
max_iter = 50
learn_rate = 0.1
sub_data = np.zeros((20, 3))  #w1,w3的数据单独拿出来
sub_data[0:10, :] = data[idx_0, :]
sub_data[10:20, :] = data[idx_3, :]
sub_label = np.zeros((20, 1))
sub_label[0:10] = 0 #重新给类别标记
sub_label[10:20] = 1
ind = [i for i in range(len(sub_label))]
random.shuffle(ind) # 随机打乱顺序
sub_data = sub_data[ind]
sub_label = sub_label[ind]
threshold = 0.5


def logistic_function(x, y):
    result = np.matmul(x, y)
    res = [math.exp(re) for re in result]
    return [1 / (1 + r) for r in res]

index = 0
for i in range(max_iter):
    predict = [round(re) for re in (logistic_function(sub_data, w))]
    error = [x - y for x, y in zip(predict, sub_label)]
    bias = learn_rate * np.matmul(sub_data.T, error)
    w = [x - y for x, y in zip(w, bias)]
    index += 1
    print(u"第%d次调整后的权值：" % index)
    print(w)
    print(u"error：%f" % np.mean(bias))
    if abs(bias[0]) < threshold and abs(bias[1])<threshold and abs(bias[2])<threshold:
        break
print("最后的weight:", w)
print('testing...')
for xn in sub_data:
    xn = xn.reshape(3,1)
    test_pre = round(logistic_function(xn.T,w)[0])
    print(test_pre)
print(sub_label)

# plt.figure(2)
# p0 = plt.scatter(data[idx_0, 0], data[idx_0, 1], marker='*', color='m', label='0', s=30)
# p3 = plt.scatter(data[idx_3, 0], data[idx_3, 1], marker='x', color='b', label=3, s=15)
# x = np.linspace(-8,0.4,8)
# y = np.linspace(0,0.4,8)
# line_cl = x*w[0] +y*w[1]+w[2]
# plt.plot(line_cl,color = 'r')
# plt.show()
