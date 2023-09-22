# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import os 
####################################################################
# 问题域相关数据读取与生成
####################################################################


# 读取数据集中的原始数据 资产 均值 标准差 协方差  资产总量=数据集大小
def read_file(prob_num):
    abs_file_path = "/home/aaa/ML/paper/APG-SMOEA/code/benchmarks/port" + str(prob_num) + ".txt"
    df = pd.read_csv(abs_file_path, header=None,
                     delimiter="\s+", names=range(3))  # info on assets 路径，指定第几行作为列名，指定分隔符，指定列名

    n = int(df[0][0])  # number of assets 第一行数字为资产总量   # n = 31
    r = df[1: (n + 1)][0].values.reshape(n, 1)  # mean of returns r->[225, 1] 第一列   # array.shape=(31,  1)
    s = df[1: (n + 1)][1].values.reshape(n, 1)  # std. of returns s->[225, 1] 第二列   # array.shape=(31,  1)
    df = df.values
    c = np.zeros((n, n))         # array.shape=(31, 31)
    for it in np.arange(n + 1, len(df)):  #   #32-528
        i, j = int(df[it][0] - 1), int(df[it][1] - 1)
        
        c[i][j] = c[j][i] = df[it][2]  # covariance between asset i, j 协方差
    return n, r, s, c   # 返回资产的个数，均值，标准差，协方差


# 根据个体构造[return, risk]  ,  也就是根据解x来进行计算指标M,V =(return, risk)
def evaluate(x, r, s, c):  # x,r,s的形状=(31,1)    r:mean of returns     s:std of returns   c的形状=(31,31) c是协方差矩阵
    M = - np.sum(np.dot(x.T, r))  # obj. 1: -1 * mean as return 收益定义 向量乘积 行*列得数   # 将每个资产的回报率乘以相应的投资比例，然后求和，最后取相反数。  取相反数的原因： 因为大多数数学优化算法都是针对最小化问题设计的。
    V = np.sum(np.dot(x, x.T) * np.dot(s, s.T) * c)  # obj. 2: variance as risk 风险定义
    return M, V   # M和V都是一个实数，越小越好，因为这里的M其实是-return ,所以是越小越好


# 读取数据集的PF面的[return, risk]       PF: pareto font
def pf(prob_num):
    abs_file_path = "/home/aaa/ML/paper/APG-SMOEA/code/benchmarks/portef" + str(prob_num) + ".txt"
    pf = np.genfromtxt(abs_file_path)  # points on pf     array.shape=(2000,2)
    M = []   # 文本文件中的数据包含多个不同解的目标函数值，包括两个目标函数值，分别是"return"（回报）和"risk"（风险）。
    V = []   # 注意，文本文件中，只是包含不同解的目标函数值，并不包含对应的解. 也就是portef1.txt中包含了2000个目标函数值，这些目标函数值可能会存在相同的解
    for i in range(len(pf)):
        M += [pf[i][0]]   # 类似于： M=[0.010865, 0.0108609579, 0.0108569167, ...]    len(M)=2000
        V += [pf[i][1]]   # len(V)=2000
    return M, V   # MV都是一维列表


# 设定问题域的主方法
def set_problem(instance):
    n, r, s, c = read_file(instance)  # 得到总资产=31    均值.shape=(31,1)  标准差.shape=(31,1) 协方差.shape=(31, 31)
    lb, ub = np.zeros((n, 1)), np.ones((n, 1))  # 设置上下界->[31, 1]   n行1列全0，n行1列全1
    port = evaluate  # port代表传入的函数名称evaluate
    mp, vp = pf(instance)   #包含不同解的目标函数值, return  risk 
    return n, r, s, c, lb, ub, port, mp, vp

'''
n: 总资产数   31
r: 均值   (31, 1)
s: 标准差  
c: 协方差
lb:每个资产的下界(lower bound)
ub: 上界(upper bound)
port = evaluate  # port代表传入的函数名称
[mp, vp]: 包含不同解的目标函数值, return  risk 
'''

# lb, ub = np.zeros((n, 1)), np.ones((n, 1))：这两行代码初始化了lb和ub，
# 它们分别表示问题中每个资产的下界（lower bound）和上界（upper bound）。
# 通常，这些边界用于限制投资比例，确保它们在0到1之间。lb初始化为0，
# 表示不允许对应资产的投资比例低于0，而ub初始化为1，表示不允许对应资产的投资比例高于1。