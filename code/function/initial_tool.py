# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# 框架初始化和评估使用
####################################################################


# 初始化邻居->B[T]
def neighbor(N, t):
    n = []
    for ni in range(N):
        D = []
        for nj in range(N):
            d = np.abs(ni - nj)
            D += [d] #99-0的数组
        index = sorted(np.arange(N), key=lambda x: D[x])
        n += [index[:int(t)]] #随机打乱的100*20的二维数组  代表每个个体邻居的编号
    return n


# 辅助-根据上下界生成均匀分布的矩阵x->[225,1]
def solution(lb, ub):
    x = []
    for l, u in zip(lb, ub):
        xi = np.random.uniform(l, u)   #从0-1中随机抽取数
        x.append(xi)
    x = np.array(x).reshape(len(x), 1)  #
    s = np.sum(x)
    if s != 0:
        x = x / s   #  ？单位化
    else:
        x = solution(lb, ub)
    return x


# 初始化种群p->[100, 225]
def population(lb, ub, N):
    P = []
    for _ in range(N):
        P.append(solution(lb, ub))
    return P   #100个array数组


# 初始化目标函数objs->[100, 2]
def objective(P, f, r, s, c):
    objs = []
    for xi in P:   #这里xi是一个含31个元素的array
        M, V = f(xi, r, s, c)  # f函数传入的是evaluate 计算return和risk 计算每个array的return和risk
        objs.append(np.array([M, V]))
    return objs

