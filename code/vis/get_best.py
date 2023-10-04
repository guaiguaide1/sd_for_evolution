import os
import numpy as np
####################################################################
# Retrieve runs with best IGD 获取最佳的IGD
####################################################################


# 获取最佳的IGD指标-min
def get_best(market, alg):
    f = open("../result/{}/{}/igd.txt".format(market, alg), "r")
    last = ""
    igd_lst = []
    for line in f:# 遍历文件的每一行
        curr = str(line).rstrip()  # 除当前行的末尾的空白字符
        if curr[:5] == "Start" and last != "" and last[0] != "=":
            # 从上一行last中的第5个字符开始提取字符串，并将其转换为浮点数，然后添加到igd_lst列表中
            igd_lst.append(float(last[4:])) 
        last = curr # 将当前行的值赋给last，为下一次循环迭代做准备。
    igd_lst.append(float(last[4:]))
    return np.argmin(igd_lst) + 1  # 使用numpy的argmin函数找到igd_lst列表中的最小值的索引，并返回其值加1（因为索引从0开始）。


# markets = ["hangseng", "dax", "ftse", "sp", "nikkei"]
markets = ["sp"]

# algs = ["de", "dem", "ga", "lvx", "adjlvx", "lvxm", "adjlvxm", "nsga2", "norm", "unif"]  # 改为10个算子
algs = ["adjlvxm", "dem", "de", "ga", "nsga2", "apg", "diff"]
for i, market in enumerate(markets):
    cmd0 = "mkdir ./datasets/{}".format(market)
    cmd1 = "mkdir ./datasets/{}/final_pop ./datasets/{}/igd".format(market, market)  # create folders
    os.system(cmd0)
    os.system(cmd1)
    for alg in algs:
        best = get_best(market, alg)  # get best runtime no.
        # copy best population file to final_pop folder
        cmd2 = "cp ../result/{}/{}/{}.csv ./datasets/{}/final_pop/{}.csv".format(market, alg, best, market, alg)
        os.system(cmd2)

        # copy best igd fraction to igd folder
        f = open("../result/{}/{}/igd.txt".format(market, alg), "r")
        w = open("./datasets/{}/igd/{}.csv".format(market, alg), "a")
        run_no = 0
        igd_frac = []
        for line in f:
            curr = str(line).rstrip()
            if curr[:5] == "Start":
                run_no += 1
            if run_no > best:
                break
            if run_no == best and curr[:5] != "Start":
                # igd_frac.append(float(curr[4:]))
                w.write(curr + "\n")
        w.close()

    # rename LEVY and CONST and adjlevy 改名是为了后面画图时候用！
    # cmd3 = "cp ./{}/final_pop/lvx.csv ./{}/final_pop/levy.csv".format(market, market)
    # cmd4 = "cp ./{}/final_pop/adjlvx.csv ./{}/final_pop/adjlevy.csv".format(market, market)
    cmd5 = "cp ./datasets/{}/final_pop/de.csv ./datasets/{}/final_pop/const.csv".format(market, market)
    # os.system(cmd3)
    # os.system(cmd4)
    os.system(cmd5)
    # cmd5 = "cp ./{}/igd/lvx.csv ./{}/igd/levy.csv".format(market, market)
    # cmd6 = "cp ./{}/igd/adjlvx.csv ./{}/igd/adjlevy.csv".format(market, market)
    cmd7 = "cp ./datasets/{}/igd/de.csv ./datasets/{}/igd/const.csv".format(market, market)
    # os.system(cmd5)
    # os.system(cmd6)
    os.system(cmd7)

    # 将当前的benchmarks中的数据copy到result中 copy portef*.txt
    cmd8 = "cp ../benchmarks/portef{}.txt ./datasets/{}/portef.txt".format(i + 1, market)
    os.system(cmd8)
