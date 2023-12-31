import numpy as np
######################################################################
# 评估指标定义6个：1.GD  2.SPC  3.SPR  4.DEL  5.IGD  6.HV 两个同名文件不可删除
######################################################################


# 计算指标辅助调用
def ndset(A):
    ndsets = []
    for i in range(len(A)):
        Mi, Vi = A[i][0], A[i][1]
        is_dominated = False     
        for j in range(len(A)):
            Mj, Vj = A[j][0], A[j][1]
            if (Mi >= Mj and Vi > Vj) or (Mi > Mj and Vi >= Vj):
                is_dominated = True   #被支配
                break
        if not is_dominated:   #不被支配  是优解 保存
            ndsets.append(A[i])
    return ndsets


# 世代距离
def gd(A, M_P, V_P):
    A = ndset(A)
    M_A, V_A = [], []
    for i in A:
        M_A.append(-i[0])
        V_A.append(i[1])
    d_sum = 0
    for i in range(len(M_A)):
        d_min = 9999
        for j in range(len(M_P)):
            d = ((M_A[i] - M_P[j]) ** 2 + (V_A[i] - V_P[j]) ** 2) ** 0.5
            if d < d_min:
                d_min = d
        d_sum += d_min
    d_avg = d_sum / len(M_A)
    return d_avg


# 反转世代距离   PF面上的解和最优解集之间的最小欧式距离
def igd(A, M_P, V_P):
    A = ndset(A)   #得到最优解集
    M_A, V_A = [], []
    for i in A:
        M_A.append(-i[0])
        V_A.append(i[1])
    d_sum = 0
    for i in range(len(M_P)):
        d_min = 9999
        for j in range(len(M_A)):
            d = ((M_P[i] - M_A[j]) ** 2 + (V_P[i] - V_A[j]) ** 2) ** 0.5  #遍历PF中的元素与最优集的元素 计算每个最优集中元素到PF元素的最小欧式距离  √￣（x²+y²）
            if d < d_min:
                d_min = d
        d_sum += d_min    
    d_avg = d_sum / len(M_P)  #求平均值
    return d_avg


# 反转世代距离 IGD+
def igd_plus(A, M_P, V_P):
    A = ndset(A)
    M_A, V_A = [], []
    for i in A:
        M_A.append(-i[0])
        V_A.append(i[1])
    d_sum = 0
    for i in range(len(M_P)):
        d_min = 9999
        for j in range(len(M_A)):
            # 修改计算公式 使用igd+指标
            d = (max((M_P[i] - M_A[j]), 0) ** 2 + max((V_P[i] - V_A[j]), 0) ** 2) ** 0.5
            if d < d_min:
                d_min = d
        d_sum += d_min
    d_avg = d_sum / len(M_P)
    return d_avg


# 空间
def spacing(A):
    A = ndset(A)
    if len(A) < 2:
        return 0
    d_lst = []
    for i in range(len(A)):
        d_min = 9999
        for j in range(len(A)):
            if i != j:
                d_ij = np.sum(np.abs(A[i] - A[j]))
                d_min = min(d_min, d_ij)
        d_lst.append(d_min)
    d_mean = np.mean(d_lst)
    s = 0
    for di in d_lst:
        s += (di - d_mean) ** 2
    SP = (s / (len(A))) ** 0.5
    return SP


# 伸展度 MaxSpread
def spread(A):
    # NOTICE: this is for Maximum Spread metric in the paper
    A = ndset(A)
    M, V = [], []
    for obj in A:
        M.append(-obj[0])
        V.append(obj[1])
    s = ((max(M) - min(M)) ** 2 + (max(V) - min(V)) ** 2) ** 0.5
    return s


# delta
def delta(A, F, L):
    # NOTICE: this is for Spread metric in the paper
    A = ndset(A)
    A = sorted(A, key=lambda x: x[1])
    M, V = [], []
    for obj in A:
        M.append(-obj[0])
        V.append(obj[1])
    df = ((M[0] - F[0]) ** 2 + (V[0] - F[1]) ** 2) ** 0.5
    dl = ((M[-1] - L[0]) ** 2 + (V[-1] - L[1]) ** 2) ** 0.5
    if len(A) > 1:
        dlist = []
        for i in range(len(A) - 1):
            di = ((M[i] - M[i + 1]) ** 2 + (V[i] - V[i + 1]) ** 2) ** 0.5
            dlist.append(di)
        d_mean = np.mean(dlist)
        temp = 0
        for di in dlist:
            temp += np.abs(di - d_mean)
    else:
        temp = 0
        d_mean = 0
    D = (df + dl + temp) / (df + dl + (len(A) - 1) * d_mean)
    return D


# 超体积HV
def hypervolume(A, R):
    A = ndset(A)
    A = sorted(A, key=lambda x: x[1])
    M, V = [], []
    for obj in A:
        M.append(-obj[0])
        V.append(obj[1])
    HV = (R[1] - V[0]) * (M[0] - R[0])
    for i in range(1, len(A)):
        HV += (R[1] - V[i]) * (M[i] - M[i - 1])
    return HV

