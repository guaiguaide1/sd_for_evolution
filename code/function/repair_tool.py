# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# 子代修正使用
####################################################################


# 辅助-根据上下界生成均匀分布的矩阵->[31,1]
def solution(lb, ub):  # lb=(31,1)
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
        x = solution(lb, ub) 
    return x


# 修正子代函数
def repair(y, lb, ub):  # 刚传进来的y.shape=(31, 1), max=1348.03   min=-0.007
    for i in range(len(y)):
        y[i] = max(y[i], lb[i])  #负数变为0, 
    s = np.sum(y)   
    if s != 0:
        y = y / s    #缩放使所有变量总和=1,  归一化操作
    else:
        # print("Repair Error: all components are 0!")
        y = solution(lb, ub)  #随机生成y  根据上下界生成均匀分布的解(31,1)
    return y
