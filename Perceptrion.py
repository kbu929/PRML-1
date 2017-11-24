import numpy as np
import random

# training data,the first four are the first class,the rest is the second categorg
X_train = [[0, 0, 0],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [0, 0, 1],
           [0, 1, 1],
           [0, 1, 0],
           [1, 1, 1]]
for i in range(len(X_train)):
    print(X_train[i])
    X_train[i].insert(0, 1)  # 在每个数组前加入1

print(X_train)
label = [1, 1, 1, 1, -1, -1, -1, -1]
c = list(zip(X_train, label))  # 打乱数据和标签的顺序
random.shuffle(c)
X_train[:], label[:] = zip(*c)
print(X_train)
print(label)
learningRate = 0.05
X_train = np.array(X_train)


def my_perceptron(train_data, label, learningRate, numOfepoch):
    w = np.zeros(len(train_data[0]))
    for epoch in range(0, numOfepoch):
        sum_error = 0.0
        for x in range(len(train_data)):
            result = np.dot(np.transpose(w), train_data[x] * label[x])
            if result <= 0:
                flag = False
                w = w + learningRate * train_data[x] * label[x]
        print(w)
    return w


w_last = my_perceptron(X_train, label, learningRate, 20)
print(w_last)
