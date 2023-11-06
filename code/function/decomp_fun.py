# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# NBI-切比雪夫分解方法
####################################################################


# 找到极值点返回;    这个函数用于从一个多目标优化问题的解集中找到最小收益和最小风险所对应的解（极值点），以便在多目标优化中进行进一步的分析或决策。
def extreme_point(obj):  #obj的形状为(100, 2)   代表把种群里面的所有解都求一遍(return, risk)
    F1 = min(obj, key=lambda x: x[0])  # lambda函数单行定义取x[0] 第一列最小值 收益最小值 F1,F2是一个包含return，risk的数组    # F1 包含了收益最小的解。
    F2 = min(obj, key=lambda x: x[1])  #第二列最小值 风险最小值   # F2 包含了最小风险和对应的收益。
    return F1, F2      # F1=array([-0.00401471,  0.00114329])      F2=array([-0.0035831 ,  0.00099589])


# 更新极值点
def update_extreme(obj, F1, F2):
    F1 = min(obj, F1, key=lambda x: x[0]) #判断生成的子代的return and risk 与原极值点的大小 更新极值点
    F2 = min(obj, F2, key=lambda x: x[1])
    return F1, F2  


# 计算权重向量返回   ？
def weight_vector(F1, F2):  # F1=array([-0.00401471,  0.00114329])  F2=array([-0.00401471,  0.00114329])
    w1 = np.abs(F1[1] - F2[1])
    w2 = np.abs(F1[0] - F2[0])
    W = np.array([w1, w2])   # W=array([0.0001474, 0.0004316])
    return W


# 生成理想点返回
def utopia_points(N, F1, F2):
    z = []
    for i in range(1, N + 1):
        ai = (N - i) / (N - 1)
        zi = ai * F1 + (1 - ai) * F2  # 通过线性组合，将 F1 和 F2 按照权重 ai 和 1 - ai 进行组合,得到乌托邦点zi
        z.append(zi)

    return z  # z.shape= (100,2)


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
        