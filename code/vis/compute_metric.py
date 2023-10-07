import numpy as np
import pandas as pd
from def_metric import hypervolume as hv
from def_metric import igd, gd
from def_metric import spacing as spc
from def_metric import spread as spr
from def_metric import delta as delta
####################################################################
# 计算指标：1.GD  2.SPC  3.SPR  4.DEL  5.IGD  6.HV
####################################################################


# markets = ["hangseng", "dax", "ftse", "sp", "nikkei"]
markets = ["hangseng"]

# algs = ["lvxm", "adjlvxm", "dem", "de", "ga", "nsga2", "lvx", "adjlvx", "unif", "norm", "de"]   #  为什么这里面有俩de
# alg_names = ["MOEA/D-Levy", "MOEA/D-AEE", "MOEA/D-DEM", "MOEA/D-DE", "MOEA/D-GA",
            #  "NSGA-II", "LEVY", "ADJLEVY", "UNIF", "NORM", "CONST"]

algs = ["adjlvxm", "dem", "de", "ga", "nsga2", "apg", "diff"]
alg_names = ["MOEA/D-AEE", "MOEA/D-DEM", "MOEA/D-DE", "MOEA/D-GA", "NSGA-II", "APG-SMOEA", "DIFF-SMOEA"]
for prob_num, market in enumerate(markets):
    pf = np.genfromtxt("../benchmarks/portef" + str(prob_num + 1) + ".txt")  # points on pf
    M, V = [], []
    for i in range(len(pf)):
        M.append(pf[i][0])
        V.append(pf[i][1])

    # compute extreme points
    F = np.array([M[np.argmin(V)], min(V)])
    L = np.array([max(M), V[np.argmax(M)]])

    # compute reference point
    R = np.array([min(M), max(V)])
    for alg in algs:
        for i in range(51):
            df = pd.read_csv("../result/{}/{}/{}.csv".format(market, alg, i + 1),
                             delimiter=",", index_col=None)
            for j in df.values:
                if -j[0] < R[0]:
                    R[0] = -j[0]
                if j[1] > R[1]:
                    R[1] = j[1]
    print("Reference point on {}: ({:.3e},{:.3e})".format(market, R[0], R[1]))

    # collect metrics for each algorithm
    GD, DEL, SPC, SPR, IGD, HV = [], [], [], [], [], []
    for alg in algs:
        GD_temp, DEL_temp, SPC_temp, SPR_temp, IGD_temp, HV_temp = [], [], [], [], [], []
        for i in range(51):
            df = pd.read_csv("../result/{}/{}/{}.csv".format(market, alg, i + 1),
                             delimiter=",", index_col=None)
            A = list(df.values)
            GD_temp.append(gd(A, M, V))
            SPC_temp.append(spc(A))
            SPR_temp.append(spr(A))
            DEL_temp.append(delta(A, F, L))
            IGD_temp.append(igd(A, M, V))
            HV_temp.append(hv(A, R))
        GD.append(GD_temp)
        DEL.append(DEL_temp)
        SPC.append(SPC_temp)
        SPR.append(SPR_temp)
        IGD.append(IGD_temp)
        HV.append(HV_temp)

    # convert into DataFrame
    metrics = [GD, DEL, SPC, SPR, IGD, HV]
    metric_names = ["GD", "Delta", "Spacing", "MaxSpread", "IGD", "Hypervolume"]
    for metric, metric_name in zip(metrics, metric_names):
        metric = pd.DataFrame(metric).T
        metric.columns = alg_names
        metric.to_csv("./num_res/{}.{}.csv".format(market, metric_name), index=False)
