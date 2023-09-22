# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# 子代修正使用
####################################################################


# 辅助-根据上下界生成均匀分布的矩阵->[225,1]
def solution(lb, ub):
    x = []
    for l, u in zip(lb, ub):
        xi = np.random.uniform(l, u)
        x.append(xi)
    x = np.array(x).reshape(len(x), 1)
    s = np.sum(x)
    if s != 0:
        x = x / s
    else:
        # print("Repair Error: all components are 0!")
        x = solution(lb, ub)  # 不成功就再来...
    return x


# 修正子代函数
def repair(y, lb, ub):
    for i in range(len(y)):
        y[i] = max(y[i], lb[i])  #负数变为0
    s = np.sum(y)   
    if s != 0:
        y = y / s    #缩放使所有变量总和=1
    else:
        # print("Repair Error: all components are 0!")
        y = solution(lb, ub)  #随机生成y  ？之前的工作无意义
    return y
