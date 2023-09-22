# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
####################################################################
# 问题域相关数据读取与生成
####################################################################


# 读取数据集中的原始数据 资产 均值 标准差 协方差  资产总量=数据集大小
def read_file(prob_num):
    df = pd.read_csv("benchmarks/port" + str(prob_num) + ".txt", header=None,
                     delimiter="\s+", names=range(3))  # info on assets 路径，指定第几行作为列名，指定分隔符，指定列名
    n = int(df[0][0])  # number of assets 第一行数字为资产总量
    r = df[1: (n + 1)][0].values.reshape(n, 1)  # mean of returns r->[225, 1] 第一列
    s = df[1: (n + 1)][1].values.reshape(n, 1)  # std. of returns s->[225, 1] 第二列
    df = df.values
    c = np.zeros((n, n))  # c->[225, 225]   
    for it in np.arange(n + 1, len(df)):  # 226-25651   #31-528
        i, j = int(df[it][0] - 1), int(df[it][1] - 1)
        
        c[i][j] = c[j][i] = df[it][2]  # covariance between asset i, j 协方差
    return n, r, s, c


# 根据个体构造[return, risk]
def evaluate(x, r, s, c):
    M = - np.sum(np.dot(x.T, r))  # obj. 1: -1 * mean as return 收益定义 向量乘积 行*列得数
    V = np.sum(np.dot(x, x.T) * np.dot(s, s.T) * c)  # obj. 2: variance as risk 风险定义
    return M, V


# 读取数据集的PF面的[return, risk]
def pf(prob_num):
    pf = np.genfromtxt("benchmarks/portef" + str(prob_num) + ".txt")  # points on pf
    M = []
    V = []
    for i in range(len(pf)):
        M += [pf[i][0]]
        V += [pf[i][1]]
    return M, V


# 设定问题域的主方法
def set_problem(instance):
    n, r, s, c = read_file(instance)  # 得到总资产 均值 标准差 协方差
    lb, ub = np.zeros((n, 1)), np.ones((n, 1))  # 设置上下界->[225, 1]   n行1列全0，n行1列全1
    port = evaluate  # port代表传入的函数名称
    mp, vp = pf(instance)
    return n, r, s, c, lb, ub, port, mp, vp
