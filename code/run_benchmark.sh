#!/usr/bin/env bash

benchmarks=("hangseng" "dax" "ftse" "sp" "nikkei")
# 分别做五个测试集上的两个实验对比测试
mkdir -p ./result/${benchmarks[$1-1]}
mkdir -p ./result/${benchmarks[$1-1]}/GAN-adjlvxm
mkdir -p ./result/${benchmarks[$1-1]}/lvxm
mkdir -p ./result/${benchmarks[$1-1]}/adjlvxm
mkdir -p ./result/${benchmarks[$1-1]}/dem
mkdir -p ./result/${benchmarks[$1-1]}/de
mkdir -p ./result/${benchmarks[$1-1]}/ga
mkdir -p ./result/${benchmarks[$1-1]}/nsga2
mkdir -p ./result/${benchmarks[$1-1]}/adjlvx
mkdir -p ./result/${benchmarks[$1-1]}/lvx
mkdir -p ./result/${benchmarks[$1-1]}/unif
mkdir -p ./result/${benchmarks[$1-1]}/norm

# 控制选择的10个测试种类 将print的结果记录到igd文件
#python run_only_GAN.py $1 >> ./result/${benchmarks[$1-1]}/GAN-adjlvxm/igd.txt
#echo Finish GAN-adjlvxm on $1

# python run_GAN-adjlvxm.py $1 >> ./result/${benchmarks[$1-1]}/GAN-adjlvxm/igd.txt    # 之前这一行xwf
python run_APG-SMOEA.py $1 >> ./result/${benchmarks[$1-1]}/GAN-adjlvxm/igd.txt
# python run_APG-SMOEA.py 1 >> ./result/hangseng/GAN-adjlvxm/igd.txt

echo Finish GAN-adjlvxm on $1

#python run_lvxm.py $1 >> ./result/${benchmarks[$1-1]}/lvxm/igd.txt
#echo Finish MOEA/D-Levy on $1
# 加入测试
#python run_adjlvxm.py $1 >> ./result/${benchmarks[$1-1]}/adjlvxm/igd.txt
#echo Finish MOEA/D-ADJLevy on $1

#python run_dem.py $1 >> ./result/${benchmarks[$1-1]}/dem/igd.txt
#echo Finish MOEA/D-DEM on $1

#python run_de.py $1 >> ./result/${benchmarks[$1-1]}/de/igd.txt
#echo Finish MOEA/D-DE and CONST on $1

#python run_ga.py $1 >> ./result/${benchmarks[$1-1]}/ga/igd.txt
#echo Finish MOEA/D-GA on $1

#python run_nsga2.py $1 >> ./result/${benchmarks[$1-1]}/nsga2/igd.txt
#echo Finish NSGA-II on $1

#python run_lvx.py $1 >> ./result/${benchmarks[$1-1]}/lvx/igd.txt
#echo Finish LEVY on $1
# 加入测试
#python run_adjlvx.py $1 >> ./result/${benchmarks[$1-1]}/adjlvx/igd.txt
#echo Finish ADJLEVY on $1

#python run_norm.py $1 >> ./result/${benchmarks[$1-1]}/norm/igd.txt
#echo Finish NORM on $1

#python run_unif.py $1 >> ./result/${benchmarks[$1-1]}/unif/igd.txt
#echo Finish UNIF on $1

