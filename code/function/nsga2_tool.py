# -*- coding: UTF-8 -*-
import numpy as np
####################################################################
# NSGA-2 工具使用
####################################################################


# 辅助-返回return, risk
def transfer_objectives(objs):
    V, M = [], []
    for obji in objs:
        Mi, Vi = obji[0], obji[1]
        M.append(Mi)
        V.append(Vi)
    return M, V


# 判断是否支配
def is_dominated(p, q, M, V): 
    Mp, Mq, Vp, Vq = M[p], M[q], V[p], V[q]
    if (Mp < Mq and Vp <= Vq) or (Mp <= Mq and Vp < Vq): #p支配q
        return 1
    else:
        return 0


def non_dominated(objs):
    M, V = transfer_objectives(objs)
    length = len(M)
    Fi = []
    for p in range(length):
        np = 0
        for q in range(length):
            if is_dominated(q, p, M, V):
                np += 1
        if np == 0:
            Fi.append(p)
    return Fi


# 快速非支配排序
def fast_non_dominated_sort(objs): #  objs.shape=(100, 2)代表把种群里面的初始所有解都求一遍(-return, risk)
    M, V = transfer_objectives(objs)  # 分别获得-return 和risk
    length = len(M)
    S, n = [], []
    rank = [0] * length  # rank=[0,0,0,...]   len(rank)=100
    F, Fi = [], []
    for p in range(length):  # length=100
        Sp = []
        np = 0
        for q in range(length):
            if is_dominated(p, q, M, V):  # 判断p是否支配q, 即是否p优于q
                Sp.append(q)
            elif is_dominated(q, p, M, V):  # 判断是否q支配p,   即q是否优于p
                np += 1
        if np == 0:   # 如果 `np == 0`，说明解 `p` 是非支配的，并将其排在第一名
            rank[p] = 1
            Fi.append(p)  # Fi: 存储非支配解的索引，即记录"最优的学生的编号"
        S.append(Sp) # S 存储每个解支配的其他解的索引
        n.append(np) # n 存储每个解被多少解支配的个数
    F.append(Fi)  # F=[[1, 7, 16, 61, 68, 78, 90]]
    i = 1
    while F[i - 1]:  # 如果存在  非支配解，即最优的学生
        Q = []
        for p in F[i-1]:# F[0]=[1, 7, 16, 61, 68, 78, 90]
            for q in S[p]:   # 进一步对 非支配解 进行排序，即对“最优的学生”再次进行排序
                n[q] -= 1
                if n[q] == 0:   # 依次找到，只受一个解支配的，只受两个解支配的，只受三个解支配的
                    rank[q] = i + 1  #  `rank[p]` 存储解 `p` 的非支配排序等级。
                    Q.append(q)
        i += 1
        F.append(Q)
    F.remove([])
    return F, rank   # rank=[8, 1, 8, 4, 11, 5, 3, 1, 3, 4, 11, 4, 5, 6, ...]  rank里面的排名，1表示不受任何解支配的，2表示只受一个解支配的，3表示只受2个解支配的
# F = [[1, 7, 16, 61, 68, 78, 90], [43, 94, 69, 98, 21, 29, 47, 22, 31, ...],...] F 中的是不同等级的解的index的集合，比如[1, 7, 16, 61, 68, 78, 90]就表示不受任何解支配的index

# 计算拥挤度
def crowding_dist_assignment(Fi, objs, dist):
    M, V = transfer_objectives(objs)
    length = len(Fi)
    for i in Fi:
        dist[i] = 0
    Fi = sorted(Fi, key=lambda x: M[x])
    dist[Fi[0]] = dist[Fi[-1]] = 9999
    for i in range(1, length-1):
        dist[Fi[i]] = dist[Fi[i]] + (np.abs(M[Fi[i+1]] - M[Fi[i-1]])) / (max(M) - min(M))
    Fi = sorted(Fi, key=lambda x: V[x])
    dist[Fi[0]] = dist[Fi[-1]] = 9999
    for i in range(1, length-1):
        dist[Fi[i]] = dist[Fi[i]] + (np.abs(V[Fi[i+1]] - V[Fi[i-1]])) / (max(V) - min(V))
    return dist
