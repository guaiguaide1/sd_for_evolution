import numpy as np
import pandas as pd
from function.prob_model import set_problem
from function.decomp_fun import extreme_point
from function.decomp_fun import update_extreme
from function.decomp_fun import utopia_points
from function.decomp_fun import weight_vector
from function.decomp_fun import to_update
from function.initial_tool import neighbor
from function.initial_tool import population
from function.initial_tool import objective
from function.nsga2_tool import fast_non_dominated_sort 
from function.repair_tool import repair
from function.GAN_model import GAN
import sys
import random

from def_metric import igd

discussion = True

def update_alpha(population, alpha_list, scaling_factor=1, avg_entropy=0):  

    
    N = len(population)
    p = np.zeros(N)  

   
    for i in range(N):

        p[i] = np.sum(np.abs(population[i]-population) ** 2)  
        p[i] = np.exp(-p[i] / (2 * np.var(p)))
       
        p[i] /= np.sum(p)  

 
    entropy = -np.sum(p * np.log2(p))

  
    if entropy < avg_entropy:
       
        alpha = max(alpha_list[-1] / scaling_factor, 0)
    else:
      
        alpha = min(alpha_list[-1] * scaling_factor, 1)

 
    alpha = alpha + 0.1 * (alpha_list[0] - alpha_list[-1])  
    alpha = max(min(alpha, 1), 0)  
    alpha = round(alpha, 4)

    return alpha, entropy
    
def data_format_transform(P, d):  # P.shape=(100, 31, 1) , d=31

    P_100 = []
    for i in range(len(P)):
        x = []
        for j in range(d):
            data = P[i][j][0]
            x.insert(j, data)
        x = np.array(x)
        P_100.append(x)
    P_100 = np.array(P_100)
    return P_100   # P_100.shape=(100, 31)
    
def data_format_recover(off, d):
    off_100 = []
    for i in range(len(off)):
        off_100.append(off[i])
    return off_100
def reshape_off(y, _):
    z = []
    for i in range(_):
        x = [y[i]]
        z.append(x)
    z = np.array(z)
    z.reshape(_, 1)
    return z

def array_merge(F):  # 数组融合，F = [[1, 7, 16, 61, 68, 78, 90], [43, 94, 69, 98, 21, 29, 47, 22, 31, ...],...] F 中的是不同等级的解的index的集合，比如[1, 7, 16, 61, 68, 78, 90]就表示不受任何解支配的index
    index = []       # 则最终的index = [1, 7, 16, 61, 68, 78, 90, 43, 94, 69, 98, 21, 29, 47, 22, 31, ...]
    for i in range(len(F)):
        index = index + F[i]
    return index

def optimize(instance, N, T, gen, operator, name, num, par, sigma, nr, cflag, cgen):  # N是种群大小, N=100   num=0,迭代的次数，即第几次迭代
    t, count, temp = 0, 0, 0   
    _, r, s, c, lb, ub, port, mp, vp = set_problem(instance)# _=31    [mp,vp]:包含不同解的目标函数值, return  risk ,应该是用来进行训练的？？
    B = neighbor(N, T)   # N=100:种群大小   T=20：邻居的个数      B:(100, 20)   # 初始化每个个体的邻居列表，以便后续在群体内进行交互和合作。从左到右，依次为从近到远。比如对于个体3的邻居：[3,2,4,1,5,0,6...]说明个体3离得最近，个体2是第二近，个体4是第三近
    upper = np.ones([1, _])  # (1,31)
    lower = np.zeros([1, _])  # (1, 31)
    
    P = population(lb, ub, N)  # (100, 31)初始化种群

    objs = objective(P, port, r, s, c) # # port代表传入的函数名称evaluate,评估种群内的解的效果  objs=(100, 2)代表把种群里面的初始所有解都求一遍(return, risk)

    F1, F2 = extreme_point(objs)   #求出初始所有解的(return, risk)包含负收益最小的(return, risk)=F1=[-0.00401471,  0.00114329]     包含risk最小的(return, risk) F2=[-0.0035831 ,  0.00099589]这个函数用于从一个多目标优化问题的解集中找到最小收益和最小风险所对应的解（极值点），以便在多目标优化中进行进一步的分析或决策。
    net = GAN(_, 8, 0.0001, 200, _)    # _=31
    indicator_value = igd(objs, mp, vp)   # obj=(100, 2)  mp=(2000, 1), vp=(2000,1)    igd: 指标值，是一个实数，越小越好，越小说明种群里的解得到的[return, risk]就越接近Pareto front解
    print("{}\t{}".format(t, indicator_value))
    
    k = 0
    alpha = 0.1  # Rn>alpha, use AEE to generate offspring; else, use GAN to generate offspring

    scaling_factor = 1.1
    alpha_list = []
    entropy_list = []
    alpha_list.append(0.1)
    while t < gen:   # gen=1500   指生成子代的代数
        
        for i in range(len(P)):   # len(P)=100   种群中有100个初始解
            if np.random.uniform(0, 1) < sigma:  # sigma=0.9     用于判断父母的选择，从邻居中还是从整个种群中
                b = B[i]  # 从邻居中选择父代, len(b)=20
            else:
                b = np.arange(N)   # 以整个种群作为父代,
            par[-1] = np.round(np.random.random(), 2) # par[],内含四个元素，在对子代用变异算子时会用到，par[0]=alpha,par[1]=beta,par[2]=pm,par[3]=etam 就是参数

            if random.random() > alpha:  # alpha=0.1   # Rn>alpha, use AEE to generate offspring; else, use GAN to generate offspring
                # P：整个种群    i:当前要变异的个体的索引
                y = operator(P, i, b, lb, ub, par)   #operator:  adj_lvxm  i:当前要变异的个体的索引。 y.shape=(31,1)变异后的个体
                # b: 个体 i 的邻居索引，从邻居中双随机选择一个个体，用于和i一起来进行变异操作。
            else :

                if k % 20 == 0 or k == 0:
                    label = np.zeros((N, 1))    # N=100 种群的大小  label.shape=(100,1)
                    F, rank = fast_non_dominated_sort(objs)  # 快速非支配排序 就是为了得到比较好的[return,risk]，将比较好的解进行排序  objs.shape=(100, 2)代表把种群里面的初始所有解都求一遍(return, risk)
                    index = array_merge(F)   # 数组融合，将F里层的[]去掉
                    index = index[:10]       # 取前面10个比较好的[return,risk],这个index就是对应的比较好的[return, risk]的索引
                    P_100 = data_format_transform(P, _)   # P_100.shape = (100, 31)   # _=31   就是简单进行了数据格式的转换，之前的P.shape=(100, 31, 1)
                    label[index, :] = 1   #  label.shape=(100,1)，把index中比较好的解的索引位置的label设置为1
                    ref_dec = P_100[index, :]    # 取出比较好的[return, risk]对应的 解,即种群中的个体 ,len(ref_dec)=10,  len(ref_dec[0])   ref_dec.shape=(10,31)
                    pool = ref_dec / np.tile(upper, (10, 1))  # upper=(1, 31)   tile(upper, (10,1)).shape=(10, 31)   np.tile 函数用于创建一个重复指定数组内容的新数组。具体而言，np.tile(upper, (10, 1)) 将 upper 数组沿着第一个维度（行）重复10次，并沿着第二个维度（列）重复1次。
                    pop_dec = P_100    # pop_dec是格式转换后的种群解  pop_dec.shape=(100, 31)
                    input_dec = (pop_dec - np.tile(lower, (np.shape(pop_dec)[0], 1))) / np.tile(upper - lower,(np.shape(pop_dec)[0], 1))
                    if t % 100 == 0 or t == 0:   # input_dec.shape=(100, 31)  label=(100,1)  label里面的某些位置为1表示这个index对应的解比较好
                        net.train(input_dec, label, pool)

                    off = net.generate(ref_dec / np.tile(upper, (np.shape(ref_dec)[0], 1)), N) * np.tile(upper, (N, 1))   # (100, 31)
                
                    off_100 = data_format_recover(off, _)
                    objs_100_y = objective(off_100, port, r, s, c)
                    F_y, rank_y = fast_non_dominated_sort(objs_100_y)
                    
                    index_y = array_merge(F_y)
                    k = 0
                y = off[index_y[k]]

                y = reshape_off(y, _)
                #y = repair(y, lb, ub)
                k = k+1



            if discussion and instance == 5 and num == 1:  # discussion=T  instance=1   num:迭代的次数，即第几次迭代
                len_trial = np.linalg.norm(y - P[i])

            obj_y = objective([y], port, r, s, c)[0]  # y.shape=(31,1)变异后的个体   评估变异后的个体的return,risk  obj_y=(return, risk)=array([-0.00014173,  0.00150852])
            M_y, V_y = obj_y[0], obj_y[1]
            F1, F2 = update_extreme(obj_y, F1, F2)   # 更新最优的performance 
            w = weight_vector(F1, F2)  # 权重向量 `W`，该向量表示了两个目标函数值向量 `F1` 和 `F2` 在各维度上的差异,可用于多目标优化算法中的权重分配和决策。  # W=array([0.0001474, 0.0004316])
            Z = utopia_points(N, F1, F2)  # Z.shape=(100, 2)  生成了一组乌托邦点Z, 代表了在多目标优化问题中的理想解
            update_count = 0   # 初始化一个计数器 update_count，用于跟踪成功更新的个体数量。
            np.random.shuffle(b)
            for bi in b:
                zk = Z[bi]   # 从乌托邦点集合 `Z` 中获取与个体 `bi` 相关的乌托邦点 `zk`
                obj_k = objs[bi]  # 获取个体 `bi` 的目标函数值向量 `obj_k`，其中包括收益和风险
                M_k, V_k = obj_k[0], obj_k[1]
                if to_update(M_y, V_y, M_k, V_k, w, zk): # 根据子代与理想点之间的相对重要性,来决定是否接受子代作为新的个体解。
                    P[bi] = y
                    objs[bi] = obj_y
                    update_count += 1
                if update_count >= nr:   #  nr:更新种群的上限，在生成子代后，用子代来替换种群中个体的次数,
                    break                #  也就是最多也只能更新俩父代，而且都是使用同样的子代来更新的父代

            if discussion and instance == 5 and num == 1:  # num:迭代的次数，即第几次迭代
                record_file = open('trial_len/' + name + '.csv', "a")
                record_file.write(f"{t},{len_trial},{update_count}\n")

                if name == 'de':
                    record_file = open('trial_len/const.csv', "a")
                    record_file.write(f"{t},{len_trial},{update_count}\n")
        alpha, entropy = update_alpha(P, alpha_list, scaling_factor,
                                      np.mean(entropy_list) if entropy_list else 0)
        alpha_list.append(alpha)
        entropy_list.append(entropy)

        t += 1

        if t in [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 150, 200, 300] and discussion and instance == 5:
            pd.DataFrame(objs, columns=["return", "risk"]).to_csv(
                f'pop/{name}_pop_gen_{t}.csv', index=False)

            if name == 'de':
                pd.DataFrame(objs, columns=["return", "risk"]).to_csv(
                    f'pop/const_pop_gen_{t}.csv', index=False)

        indicator_value = igd(objs, mp, vp)  # compute IGD metric
        print("{}\t{}".format(t, indicator_value))

        if np.abs(indicator_value - temp) >= 1e-05:
            temp = indicator_value
            count = 0
        else:
            count += 1
        if count >= cgen and cflag is True:
            break
    return objs
