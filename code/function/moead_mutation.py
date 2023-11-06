# -*- coding: UTF-8 -*-
import numpy as np
from scipy.special import gamma as G
from function.repair_tool import repair
####################################################################
# MOEA/D 变异使用
####################################################################


# 辅助-模拟二进制交叉算子
def sbx_crossover(p1, p2, lb, ub, etac):
    c1, c2 = [], []
    for i in range(len(p1)):
        x1 = min(p1[i], p2[i])
        x2 = max(p1[i], p2[i])
        xl, xu = lb[i], ub[i]
        if np.random.uniform(0, 1) < 0.5:
            if x1 != x2:
                myu = np.random.uniform(0, 1)
                beta1 = 1 + 2 * (x1 - xl) / (x2 - x1)
                beta2 = 1 + 2 * (xu - x2) / (x2 - x1)
                alpha1 = 2 - beta1 ** (-(etac + 1))
                alpha2 = 2 - beta2 ** (-(etac + 1))
                if myu <= 1 / alpha1:
                    betaq1 = (myu * alpha1) ** (1 / (etac + 1))
                else:
                    betaq1 = (1 / (2 - myu * alpha1)) ** (1 / (etac + 1))
                if myu <= 1 / alpha2:
                    betaq2 = (myu * alpha2) ** (1 / (etac + 1))
                else:
                    betaq2 = (1 / (2 - myu * alpha2)) ** (1 / (etac + 1))

                c1i = 0.5 * ((x1 + x2) - betaq1 * (x2 - x1))
                c2i = 0.5 * ((x1 + x2) + betaq2 * (x2 - x1))
                c1.append(c1i)
                c2.append(c2i)
            else:
                c1.append(x1)
                c2.append(x2)
        else:
            c1.append(x1)
            c2.append(x2)
    c1 = np.array(c1).reshape(len(c1), 1)
    c2 = np.array(c2).reshape(len(c2), 1)
    return c1, c2


# GA-多项式变异算子
def ga(P, i, b, lb, ub, par):
    pc, pm, etac, etam = par[0], par[1], par[2], par[3]
    np.random.shuffle(b)
    p1, p2 = P[b[0]], P[b[1]]
    if np.random.uniform(0, 1) < pc:
        c1, c2 = sbx_crossover(p1, p2, lb, ub, etac)
    else:
        c1, c2 = p1, p2
    if np.random.uniform(0, 1) < 0.5:
        y = c1
    else:
        y = c2
    y = repair(y, lb, ub)
    y = poly_mutation(y, lb, ub, etam, pm)
    y = repair(y, lb, ub)
    return y


# DE算子
def de(P, i, b, lb, ub, par):
    f = par[0]
    np.random.shuffle(b)
    p1 = P[i]
    p2, p3 = P[b[0]], P[b[1]]
    y = p1 + f * (p2 - p3)
    y = repair(y, lb, ub)
    return y


# DE均匀分布算子
def de_uniform(P, i, b, lb, ub, par):
    f = par[0]
    np.random.shuffle(b)
    p1 = P[i]
    p2, p3 = P[b[0]], P[b[1]]
    y = np.zeros((len(p1), 1))
    for j in range(len(p1)):
        y[j] = p1[j] + np.random.uniform(-f, f) * (p2[j] - p3[j])
    y = repair(y, lb, ub)
    return y


# DE正态分布算子
def de_normal(P, i, b, lb, ub, par):
    f = par[0]
    np.random.shuffle(b)
    p1 = P[i]
    p2, p3 = P[b[0]], P[b[1]]
    y = np.zeros((len(p1), 1))
    for j in range(len(p1)):
        y[j] = p1[j] + f * np.random.normal(1) * (p2[j] - p3[j])
    y = repair(y, lb, ub)
    return y


# DEM-多项式变异算子
def dem(P, i, b, lb, ub, par):
    f, pm, etam = par[0], par[1], par[2]
    np.random.shuffle(b)
    p1 = P[i]
    p2, p3 = P[b[0]], P[b[1]]
    y = p1 + f * (p2 - p3)
    y = repair(y, lb, ub)
    y = poly_mutation(y, lb, ub, etam, pm)
    y = repair(y, lb, ub)
    return y


# lvxm-多项式变异算子
def lvxm(P, i, b, lb, ub, par):
    alpha, beta, pm, etam = par[0], par[1], par[2], par[3]
    np.random.shuffle(b)   #打乱顺序
    p1 = P[i]
    p2 = P[b[0]]   #b[0]是一个随机值
    y = p1 + alpha * levy(beta, len(p1)) * (p1 - p2)
    y = repair(y, lb, ub)
    y = poly_mutation(y, lb, ub, etam, pm)
    y = repair(y, lb, ub)
    return y


# lvx算子
def lvx(P, i, b, lb, ub, par):
    alpha, beta = par[0], par[1]
    np.random.shuffle(b)
    p1 = P[i]
    p2 = P[b[0]]
    y = p1 + alpha * levy(beta, len(p1)) * (p1 - p2)
    y = repair(y, lb, ub)
    return y


##########################################################
# idea3->新算子自适应 两个实验对比使用adj_lvxm adj_lvx
# 将原有的levy-flight变异公式修改 引入参数λ做实验观察效果
# 2020-11-26 frank

# adj_lvxm-多项式变异算子(5个参数)  # par=[1e-05, 0.3, 0.03226, 20, 0.5]
def adj_lvxm(P, i, b, lb, ub, par):  # i: 当前要变异的个体的索引
    alpha, beta, pm, etam, epsilon = par[0], par[1], par[2], par[3], par[4]
    np.random.shuffle(b)
    p1 = P[i]   # （31,1)
    p2 = P[b[0]]  # 从邻居中随机选择一个个体p2   (31, 1)
    y = p1 * epsilon + alpha * levy(beta, len(p1)) * (p1 - p2) * (1 - epsilon)  # 公式8    y=（31， 1）
    y = repair(y, lb, ub)# 对变异后的个体 `y` 进行修复操作，确保它在上下界 `lb` 和 `ub` 内。
    y = poly_mutation(y, lb, ub, etam, pm)  # 对变异后的个体 `y` 进行多项式变异操作，但这次使用参数 `etam` 和 `pm` 来控制多项式变异
    y = repair(y, lb, ub)  # 再次对变异后的个体 `y` 进行修复操作，确保它在上下界 `lb` 和 `ub` 内
    return y   # （31， 1)


def enhanced_lvxm(P, i, b, lb, ub, par):
    alpha, beta, pm, etam, epsilon_init, epsilon_final, gen, max_gen = par
    epsilon = epsilon_init + (epsilon_final - epsilon_init) * (gen / max_gen)  # 动态参数调整

    np.random.shuffle(b)
    p1 = P[i]
    p2, p3, p4 = P[b[0]], P[b[1]], P[b[2]]
    
    # 差分进化
    diff_vector = alpha * (p2 - p3) + (1 - alpha) * (p4 - p1)
    
    # 使用Levy飞行生成的步长与差分进化生成的方向结合
    y = p1 + epsilon * diff_vector + (1 - epsilon) * levy(beta, len(p1)) * diff_vector
    
    y = repair(y, lb, ub)
    y = poly_mutation(y, lb, ub, etam, pm)
    y = repair(y, lb, ub)
    
    return y




def gaussian_mutation(p, mu, sigma):
    if np.random.uniform(0, 1) < 0.01:
        p = p + np.random.normal(mu, sigma, size=p.shape)
    return p

def crossover(p1, p2):
    if np.random.uniform(0, 1) < 1:
        alpha = np.random.uniform(0, 1, size=p1.shape)
        return alpha * p1 + (1 - alpha) * p2
    else:
        return p1

def adj_lvxm_improved(P, i, b, lb, ub, par):
    gamma, delta, alpha, beta, pm, etam, epsilon, mu, sigma = par
    np.random.shuffle(b)
    p1 = P[i]
    p2 = P[b[0]]
    y = p1 * epsilon + alpha * levy(beta, len(p1)) * (p1 - p2) * (1 - epsilon)
    y = repair(y, lb, ub)
    y = crossover(y, p2)
    y = poly_mutation(y, lb, ub, etam, pm)
    # y = gaussian_mutation(y, mu, sigma)
    y = repair(y, lb, ub)
    return y


def diff(p1, p2, lb, ub, par):
    alpha, beta, pm, etam, epsilon = par[0], par[1], par[2], par[3], par[4]
    y = p1 * epsilon + alpha * levy(beta, len(p1)) * (p1 - p2) * (1 - epsilon)  # 公式8    y=（31， 1）
    y = repair(y, lb, ub)# 对变异后的个体 `y` 进行修复操作，确保它在上下界 `lb` 和 `ub` 内。
    y = poly_mutation(y, lb, ub, etam, pm)  # 对变异后的个体 `y` 进行多项式变异操作，但这次使用参数 `etam` 和 `pm` 来控制多项式变异
    y = repair(y, lb, ub)  # 再次对变异后的个体 `y` 进行修复操作，确保它在上下界 `lb` 和 `ub` 内
    return y   # （31， 1)




# adj_lvx-多项式变异算子(3个参数)
def adj_lvx(P, i, b, lb, ub, par):
    alpha, beta, epsilon = par[0], par[1], par[2]
    np.random.shuffle(b)
    p1 = P[i]
    p2 = P[b[0]]
    y = p1 * epsilon + alpha * levy(beta, len(p1)) * (p1 - p2) * (1 - epsilon)
    y = repair(y, lb, ub)
    return y
##########################################################


# 辅助-多项式变异算子    多项式变异是一种用于优化算法的变异策略，它通常用于在搜索空间中引入多样性，以帮助算法探索更广泛的解空间
def poly_mutation(p, lb, ub, etam, pm):  # p.shape=(31,1)   etam=20,   pm=0.03226
    for i in range(len(p)):
        if np.random.uniform(0, 1) < pm:
            x = p[i]
            xl, xu = lb[i], ub[i]
            myu = np.random.uniform(0, 1)
            if myu < 0.5:
                sigmaq = (2 * myu) ** (1 / (etam + 1)) - 1
            else:
                sigmaq = 1 - (2 * (1 - myu)) ** (1 / (etam + 1))
            p[i] = x + sigmaq * (xu - xl)
    return p   # 多项式变异会导致个体中的某些决策变量值发生改变，从而可能破坏原来的和为1的性质; 只是随机挑选某些位置上的值进行变异，这个变异的概率比较小啦，只有0.03226


# 辅助-LEVY函数.    这个函数的目的是生成服从 Lévy 分布的随机数，通常用于优化算法中的多项式变异等操作。
def levy(beta, n):   # beta=0.3, n=len(p1)=31
    num = G(1 + beta) * np.sin(np.pi * beta / 2)
    den = G((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # 公式6
    sigma_u, sigma_v = (num / den) ** (1 / beta), 1
    u = np.random.normal(0, sigma_u, size=n)
    v = np.random.normal(0, sigma_v, size=n)
    z = u / (np.abs(v) ** (1 / beta))
    return z.reshape(n, 1)
