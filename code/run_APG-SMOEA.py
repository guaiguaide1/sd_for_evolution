# -*- coding: UTF-8 -*-
import sys
import numpy as np
import random
import pandas as pd
from function.moead_mutation import adj_lvxm
from function.APG_SMOEA_framework import optimize


instance = int(sys.argv[1])    # 在这里接受了参数，在数据集instance上进行测试  ，# instance=1   用来判断使用的数据集
rep = 51   # 迭代的次数    # 31 85 89 98 225
benchmarks = ["hangseng", "dax", "ftse", "sp", "nikkei"]   # 包含多个股市的名称
size = [31, 85, 89, 98, 225]    # 数据集大小
# savedir = "result/{}/GAN-adjlvxm/".format(benchmarks[instance-1])   # 'result/hangseng/GAN-adjlvxm/'
savedir = "result/{}/DIFF-adjlvxm/".format(benchmarks[instance-1])

N, T, gen = 100, 20, 1500 # N:数据集大小   # T:邻居大小   # gen:与迭代次数不同，这里只生成子代的代数
sigma, nr = 0.9, 2     # sigma:用于判断父母的选择，从邻居中还是从整个种群中    # 
par = [1e-05, 0.3, round(1/size[instance-1], 5), 20, 0.5  ]  # [1e-05, 0.3, 0.03226, 20, 0.5]
#  par内含四个元素，在对子代用变异算子时会用到，par[0]=alpha,par[1]=beta,par[2]=pm,par[3]=etam 就是参数
# print(instance, benchmarks[instance-1])
# print(par)
# print("====================================")

for i in range(rep): 
    np.random.seed(500+i)
    random.seed(500+i)
    print("Start {}-th experiment.".format(i+1))
    # res = optimize(instance, N, T, gen, adj_lvxm, 'APG-SMOEA', i, par, sigma, nr, True, 100, 0)
    res = optimize(instance, N, T, gen, adj_lvxm, 'DIFF-SMOEA', i, par, sigma, nr, True, 100, 1)  # lastPara=1 choose Diffusion
    res = pd.DataFrame(res, columns=["return", "risk"])
    res.to_csv(savedir + str(i+1) + ".csv", index=False)


