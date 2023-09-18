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
    
def data_format_transform(P, d):

    P_100 = []
    for i in range(len(P)):
        x = []
        for j in range(d):
            data = P[i][j][0]
            x.insert(j, data)
        x = np.array(x)
        P_100.append(x)
    P_100 = np.array(P_100)
    return P_100
    
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

def array_merge(F):
    index = []
    for i in range(len(F)):
        index = index + F[i]
    return index

def optimize(instance, N, T, gen, operator, name, num, par, sigma, nr, cflag, cgen):
    t, count, temp = 0, 0, 0
    _, r, s, c, lb, ub, port, mp, vp = set_problem(instance)
    B = neighbor(N, T)
    upper = np.ones([1, _])
    lower = np.zeros([1, _])
    
    P = population(lb, ub, N)

    objs = objective(P, port, r, s, c)

    F1, F2 = extreme_point(objs)
    net = GAN(_, 8, 0.0001, 200, _)
    indicator_value = igd(objs, mp, vp)
    print("{}\t{}".format(t, indicator_value))
    
    k = 0
    alpha = 0.1

    scaling_factor = 1.1
    alpha_list = []
    entropy_list = []
    alpha_list.append(0.1)
    while t < gen:
        
        for i in range(len(P)):
            if np.random.uniform(0, 1) < sigma:
                b = B[i]
            else:
                b = np.arange(N)
            par[-1] = np.round(np.random.random(), 2)

            if random.random() > alpha:
                
                y = operator(P, i, b, lb, ub, par)
                
            else :

                if k % 20 == 0 or k == 0:
                    label = np.zeros((N, 1))
                    F, rank = fast_non_dominated_sort(objs)
                    index = array_merge(F)
                    index = index[:10]
                    P_100 = data_format_transform(P, _)
                    label[index, :] = 1
                    ref_dec = P_100[index, :]
                    pool = ref_dec / np.tile(upper, (10, 1))
                    pop_dec = P_100
                    input_dec = (pop_dec - np.tile(lower, (np.shape(pop_dec)[0], 1))) / np.tile(upper - lower,(np.shape(pop_dec)[0], 1))
                    if t % 100 == 0 or t == 0: 
                        net.train(input_dec, label, pool)

                    off = net.generate(ref_dec / np.tile(upper, (np.shape(ref_dec)[0], 1)), N) * np.tile(upper, (N, 1))
                
                    off_100 = data_format_recover(off, _)
                    objs_100_y = objective(off_100, port, r, s, c)
                    F_y, rank_y = fast_non_dominated_sort(objs_100_y)
                    
                    index_y = array_merge(F_y)
                    k = 0
                y = off[index_y[k]]

                y = reshape_off(y, _)
                #y = repair(y, lb, ub)
                k = k+1



            if discussion and instance == 5 and num == 1:
                len_trial = np.linalg.norm(y - P[i])

            obj_y = objective([y], port, r, s, c)[0]
            M_y, V_y = obj_y[0], obj_y[1]
            F1, F2 = update_extreme(obj_y, F1, F2)
            w = weight_vector(F1, F2)
            Z = utopia_points(N, F1, F2)
            update_count = 0
            np.random.shuffle(b)
            for bi in b:
                zk = Z[bi]
                obj_k = objs[bi]
                M_k, V_k = obj_k[0], obj_k[1]
                if to_update(M_y, V_y, M_k, V_k, w, zk):
                    P[bi] = y
                    objs[bi] = obj_y
                    update_count += 1
                if update_count >= nr:
                    break

            if discussion and instance == 5 and num == 1:
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
