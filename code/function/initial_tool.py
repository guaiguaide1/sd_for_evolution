# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# 框架初始化和评估使用
####################################################################

# 初始化每个个体的邻居列表，以便后续在群体内进行交互和合作。从左到右，依次为从近到远。比如对于个体3的邻居：[3,2,4,1,5,0,6...]说明个体3离得最近，个体2是第二近，个体4是第三近
# 初始化邻居->B[T]   
def neighbor(N, t):  # N=100代表种群中个体的数量   t=20代表每个个体的邻居数量
    n = []
    for ni in range(N):
        D = []
        for nj in range(N):
            d = np.abs(ni - nj)
            D += [d] #99-0的数组
        index = sorted(np.arange(N), key=lambda x: D[x])
        n += [index[:int(t)]] #随机打乱的100*20的二维数组  代表每个个体邻居的编号,即索引  
    return n    # 二维列表  100*20   
#  比如对于个体3的邻居：[3,2,4,1,5,0,6...]说明个体3离得最近，个体2是第二近，个体4是第三近

# 辅助-根据上下界生成均匀分布的矩阵x->[31,1]
def solution(lb, ub):   # lb, ub = np.zeros((31, 1)), np.ones((31, 1))
    x = []
    for l, u in zip(lb, ub):
        xi = np.random.uniform(l, u)   #从0-1中随机抽取数
        x.append(xi)
    x = np.array(x).reshape(len(x), 1)  #(31, 1)
    s = np.sum(x)
    if s != 0:
        x = x / s    # 归一化，确保 和 为1，用于初始化投资组合中资产的初始分配比例。
    else:
        x = solution(lb, ub)
    return x


# 初始化种群p->[100, 225]   lb, ub = np.zeros((31, 1)), np.ones((31, 1))：这两行代码初始化了lb和ub，它们分别表示问题中每个资产的下界（lower bound）和上界（upper bound）。
def population(lb, ub, N):  # 种群大小N=100
    P = []
    for _ in range(N):
        P.append(solution(lb, ub))
    return P   #(100, 31)初始化种群


# 初始化目标函数objs->[100, 2]
def objective(P, f, r, s, c):
    objs = []
    for xi in P:   #这里xi是一个含31个元素的array, xi就是一个解，P是种群，即解的集合
        M, V = f(xi, r, s, c)  # f函数传入的是evaluate, 也就是根据解x来进行计算指标M,V =(-return, risk)
        objs.append(np.array([M, V]))
    return objs    # （100， 2）

# P: (100, 31)初始化种群
# f代表传入的函数名称evaluate
# r: mean of returns
# s: std of returns 
# c: 31个资产的协方差矩阵