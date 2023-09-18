# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# NBI-切比雪夫分解方法
####################################################################


# 找到极值点返回
def extreme_point(obj):
    F1 = min(obj, key=lambda x: x[0])  # lambda函数单行定义取x[0] 第一列最小值 收益最小值 F1,F2是一个包含return，risk的数组
    F2 = min(obj, key=lambda x: x[1])  #第二列最小值 风险最小值
    return F1, F2


# 更新极值点
def update_extreme(obj, F1, F2):
    F1 = min(obj, F1, key=lambda x: x[0]) #判断生成的子代的return and risk 与原极值点的大小 更新极值点
    F2 = min(obj, F2, key=lambda x: x[1])
    return F1, F2


# 计算权重向量返回   ？
def weight_vector(F1, F2):
    w1 = np.abs(F1[1] - F2[1])
    w2 = np.abs(F1[0] - F2[0])
    W = np.array([w1, w2])
    return W


# 生成理想点返回
def utopia_points(N, F1, F2):
    z = []
    for i in range(1, N + 1):
        ai = (N - i) / (N - 1)
        zi = ai * F1 + (1 - ai) * F2
        z.append(zi)

    return z


# 判断子代是否需要更新
def to_update(M_y, V_y, M_k, V_k, w, zk):
    w1, w2 = w[0], w[1]  # 权重
    z1, z2 = zk[0], zk[1]  # 理想点
    y1, y2 = M_y, V_y  # 子代
    k1, k2 = M_k, V_k  # 父母
    gy = max(w1 * (y1 - z1), w2 * (y2 - z2))
    gk = max(w1 * (k1 - z1), w2 * (k2 - z2))
    if gy <= gk:
        return True
    else:
        return False
