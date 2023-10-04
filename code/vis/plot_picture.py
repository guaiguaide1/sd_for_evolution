import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
####################################################################
# 作图 画出指标对比图
####################################################################


# read igd by generation data
def load_igd(market, alg):
    dat = np.loadtxt("./datasets/{}/igd/{}.csv".format(market, alg), delimiter="\t")
    igd = list(dat[:, 1])# 从加载的数据中提取第二列（索引为1，因为索引从0开始），并将其转化为列表
    
    # 循环，确保igd列表的长度至少为1501
    # 如果igd的长度小于1501，则会重复地添加igd的最后一个元素，直到它的长度达到1501。
    while len(igd) < 1501:
        igd.append(igd[-1])

    # 创建一个字典，其中包含两个键："Generations"和"IGD"。"Generations"的值是一个从0到1500的整数序列，而"IGD"的值是处理后的igd列表。
    dat = {"Generations": list(np.arange(1501)), "IGD": igd}
    dat = pd.DataFrame(dat)
    return dat


# read points in each algorithms solution set
def load_sol(market, alg):
    dat = pd.read_csv("./datasets/{}/final_pop/{}.csv".format(market, alg), delimiter=",")
    dat["return"] = -dat["return"]
    return dat


styles = ["paper"]
# markets = ["hangseng", "dax", "ftse", "sp", "nikkei"]
markets = ["sp"]

# xlims：可能用于表示某种图形或数据的x轴的范围
# xlims中的第一个元组(0.001, 0.0015)可能表示x轴的范围从0.001到0.0015
xlims = [(0.001, 0.0015), (0.0005, 0.0008), (0.0003, 0.0004), (0.0005, 0.001), (0.0004, 0.0008)]
ylims = [(0.006, 0.008), (0.008, 0.009), (0.004, 0.006), (0.006, 0.008), (0.0015, 0.0035)]

# set algorithm names, labels, markers and colors 共11个
# algs1 = ["lvxm", "dem", "de", "ga", "nsga2", "adjlvxm"]
# algs2 = ["levy", "unif", "norm", "const", "adjlevy"]
# labs1 = ["MOEA/D-Lévy", "MOEA/D-DEM", "MOEA/D-DE", "MOEA/D-GA", "NSGA-II", "MOEA/D-AEE"]
# labs2 = ["LEVY", "UNIF", "NORM", "CONST", "ADJLEVY"]
# maks1 = ["o", "v", "^", "s", "D", "<"]
# maks2 = ["o", "v", "^", "s", "D"]
# cols1 = ["b", "g", "r", "c", "m", "y"]
# cols2 = ["b", "g", "r", "c", "m"]
algs1 = ["adjlvxm", "dem", "de", "ga", "nsga2", "apg", "diff"]
labs1 = ["MOEA/D-AEE", "MOEA/D-DEM", "MOEA/D-DE", "MOEA/D-GA", "NSGA-II", "APG-SMOEA", "DIFF-SMOEA"]
maks1 = ["o", "v", "^", "s", "D", "<", "p"]
cols1 = ["b", "g", "r", "c", "m", "y", "k"]

for style in styles:
    if style == "paper":
        font_scale = 1.2
    else:
        font_scale = 1
    # set seaborn style
    sns.set(style, "white", "bright", font_scale=font_scale,
            rc={'font.family': ['sans-serif'],
                'font.sans-serif': ['Arial',
                                    'DejaVu Sans',
                                    'Liberation Sans',
                                    'Bitstream Vera Sans',
                                    'sans-serif'],
                'axes.edgecolor': '.0',
                'axes.labelcolor': '.0',
                'text.color': '.0',
                'xtick.bottom': True,
                'xtick.color': '.0',
                'xtick.direction': 'in',
                'xtick.top': True,
                'xtick.major.size': 3,
                'ytick.color': '.0',
                'ytick.direction': 'in',
                'ytick.left': True,
                'ytick.right': True,
                'ytick.major.size': 3, })
    for i, market in enumerate(markets):
        # 读取PF面的数据
        pf = pd.DataFrame(np.genfromtxt("./datasets/{}/portef.txt".format(market)), columns=["return", "risk"])
        dats1 = []
        for alg in algs1:
            dats1.append(load_igd(market, alg))

        # dats2 = []
        # for alg in algs2:
        #     dats2.append(load_igd(market, alg))

        # 算法对比 igd1 #################################################################
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        for dat, mak, col, lab in zip(dats1, maks1, cols1, labs1):
            plt.plot(dat["Generations"], dat["IGD"], label=lab, marker=mak, markevery=200,
                     color=col, markeredgecolor=col, markerfacecolor="none")
        plt.xlabel("Generations")
        plt.ylabel("IGD")
        plt.legend(frameon=True,
                   loc="upper right",
                   edgecolor="black",
                   fancybox=False)
        plt.savefig("./exp_images/{}_igd1_{}.jpg".format(market, style))
        plt.clf()

        # 算子对比 igd2 #################################################################
        # fig = plt.figure(figsize=(6, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # # ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
        # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # for dat, mak, col, lab in zip(dats2, maks2, cols2, labs2):
        #     plt.plot(dat["Generations"], dat["IGD"], label=lab, marker=mak, markevery=200,
        #              color=col, markeredgecolor=col, markerfacecolor="none")
        # plt.xlabel("Generations")
        # plt.ylabel("IGD")
        # plt.legend(frameon=True,
        #            loc="upper right",
        #            edgecolor="black",
        #            fancybox=False)
        # plt.savefig("./exp_images/{}_igd2_{}.jpg".format(market, style))
        # plt.clf()

        dats1 = []
        for alg in algs1:
            dats1.append(load_sol(market, alg))

        # dats2 = []
        # for alg in algs2:
        #     dats2.append(load_sol(market, alg))

        # 算法对比种群 pop1 #################################################################
        # fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.plot(pf["risk"], pf["return"], label="Pareto Front", c="black")
        for dat, mak, col, lab in zip(dats1, maks1, cols1, labs1):
            plt.scatter(dat["risk"], dat["return"], label=lab, marker=mak,
                        edgecolors=col, facecolor="none")
        plt.xlabel("Risk")
        plt.ylabel("Return")
        plt.legend(frameon=True,
                   loc="lower right",
                   edgecolor="black",
                   fancybox=False)
        plt.savefig("./exp_images/{}_pop1_{}.jpg".format(market, style))
        plt.clf()

        



        # 算子对比种群 pop2 #################################################################
        # fig = plt.figure(figsize=(6, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # plt.plot(pf["risk"], pf["return"], label="Pareto Front", c="black")
        # for dat, mak, col, lab in zip(dats2, maks2, cols2, labs2):
        #     plt.scatter(dat["risk"], dat["return"], label=lab, marker=mak,
        #                 edgecolors=col, facecolor="none")
        # plt.xlabel("Risk")
        # plt.ylabel("Return")
        # plt.legend(frameon=True,
        #            loc="lower right",
        #            edgecolor="black",
        #            fancybox=False)
        # plt.savefig("./exp_images/{}_pop2_{}.jpg".format(market, style))
        # plt.clf()



        # 算法对比种群规模 pop1scale #################################################################
        # fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.plot(pf["risk"], pf["return"], label="Pareto Front", c="black")
        for dat, mak, col, lab in zip(dats1, maks1, cols1, labs1):
            plt.scatter(dat["risk"], dat["return"], label=lab, marker=mak,
                        edgecolors=col, facecolor="none")
        plt.xlim(xlims[i])
        plt.ylim(ylims[i])
        plt.xticks([])
        plt.yticks([])
        plt.savefig("./exp_images/{}_pop1scale_{}.jpg".format(market, style))
        plt.clf()


        # 算子对比种群规模 pop2scale #################################################################
        # fig = plt.figure(figsize=(6, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # plt.plot(pf["risk"], pf["return"], label="Pareto Front", c="black")
        # for dat, mak, col, lab in zip(dats2, maks2, cols2, labs2):
        #     plt.scatter(dat["risk"], dat["return"], label=lab, marker=mak,
        #                 edgecolors=col, facecolor="none")
        # plt.xlim(xlims[i])
        # plt.ylim(ylims[i])
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig("./exp_images/{}_pop2scale_{}.jpg".format(market, style))
        # plt.clf()
