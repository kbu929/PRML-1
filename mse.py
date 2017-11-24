'''
   算法：最小均方算法(lms)
   均方误差：样本预测输出值与实际输出值之差平方的期望值，记为MES
   设:observed 为样本真值,predicted为样本预测值,则计算公式:
   (转换为容易书写的方式，非数学标准写法,因为数学符号在这里不好写)
   MES=[(observed[0]-pridicted[0])*(observed[0]-pridicted[0])+....
         (observed[n]-pridicted[n])*(observed[n]-pridicted[n])]/n
'''

'''
   变量约定：大写表示矩阵或数组，小写表示数字
   X：表示数组或者矩阵
   x:表示对应数组或矩阵的某个值
'''

'''
     关于学习效率（也叫步长：控制着第n次迭代中作用于权值向量的调节）。(下面的参数a)：
     学习效率过大：收敛速度提高，稳定性降低，即出结果快，但是结果准确性较差
     学习效率过小：稳定性提高，收敛速度降低，即出结果慢，准确性高，耗费资源
     对于学习效率的确定，有专门的算法，这里不做研究。仅仅按照大多数情况下的选择：折中值
'''
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
a = 0.1  ##学习率 0<a<1
X = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]])  ##输入矩阵
D = np.array([1, 1, 1, 1, 0, 0, 0, 0])  ##期望输出结果矩阵
W = np.array([-1, -2, -2])  ##权重向量
expect_e = 0.005  ##期望误差
maxtrycount = 20  ##最大尝试次数

##硬限幅函数(即标准,这个比较简单：输入v大于0，返回1.小于等于0返回-1)
'''
    最后的权重为W([0.1,0.1]),则:0.1x+0.1y=0 ==>y=-x
    即：分类线方程为:y=-x
'''


def sgn(v):
    if v > 0:
        return 1
    else:
        return 0  ##跟上篇感知器单样本训练的-1比调整成了0，为了测试需要。-1训练不出结果


##读取实际输出
'''
    这里是两个向量相乘，对应的数学公式：
    a(m,n)*b(p,q)=m*p+n*q
    在下面的函数中，当循环中xn=1时(此时W=([0.1,0.1]))：
    np.dot(W.T,x)=(1,1)*(0.1,0.1)=1*0.1+1*0.1=0.2>0 ==>sgn 返回1
'''


def get_v(W, x):
    return sgn(np.dot(W.T, x))  ##dot表示两个矩阵相乘


##读取误差值
def get_e(W, x, d):
    return d - get_v(W, x)


##权重计算函数(批量修正)
'''
  对应数学公式: w(n+1)=w(n)+a*x(n)*e
  对应下列变量的解释：
  w(n+1) <= neww 的返回值
  w(n)   <=oldw(旧的权重向量)
  a      <= a(学习率，范围：0<a<1)
  x(n)   <= x(输入值)
  e      <= 误差值或者误差信号
'''


def neww(oldW, d, x, a):
    e = get_e(oldW, x, d)
    return (oldW + a * x * e, e)


##修正权值
'''
    此循环的原理：
    权值修正原理(批量修正)==>神经网络每次读入一个样本，进行修正，
        达到预期误差值或者最大尝试次数结束，修正过程结束   
'''
cnt = 0
while True:
    err = 0
    i = 0
    for xn in X:
        W, e = neww(W, D[i], xn, a)
        i += 1
        err += pow(e, 2)  ##lms算法的核心步骤，即：MES
    err /= float(i)
    cnt += 1
    print(u"第 %d 次调整后的权值：" % cnt)
    print(W)
    print(u"误差：%f" % err)
    if err < expect_e or cnt >= maxtrycount:
        break

print("最后的权值：", W.T)

##输出结果
print("开始验证结果...")
for xn in X:
    print("D%s and W%s =>%d" % (xn, W.T, get_v(W, xn)))

##测试准确性：
'''
   由上面的说明可知：分类线方程为y=-x,从坐标轴上可以看出：
   (2,3)属于+1分类,(-2,-1)属于0分类
'''

# print("开始测试...")
# test = np.array([2, 3])
# print("D%s and W%s =>%d" % (test, W.T, get_v(W, test)))
# test = np.array([-2, -1])
# print("D%s and W%s =>%d" % (test, W.T, get_v(W, test)))
ax=plt.subplot(111,projection='3d')
for xn in X:
    ax.scatter(xn[0],xn[1],xn[2],c = 'y')
    plt.show()