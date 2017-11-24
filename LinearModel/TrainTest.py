import LinearModel
import numpy as np
import os
def  train_test(dataset,label,attr_count):
    batch_size = dataset//10
    for idx in range(10):
        print('十折交叉验证：当前第 %d 次'%(idx+1))

