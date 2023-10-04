#!/usr/bin/env bash

# 入口脚本！
# result生成原始结果 paper_process使用
mkdir -p ./result
# 在nikkei上的两个实验生成讨论的结果 discussion_process使用
mkdir -p ./pop
mkdir -p ./trial_len

# 数字控制跑哪些测试集bash run_all.sh
# instance序号：1-hangseng 2-dax 3-ftse 4-sp 5-nikkei
# bash ./run_benchmark.sh 1
# bash ./run_benchmark.sh 2
# bash ./run_benchmark.sh 3
bash ./run_benchmark.sh 4
# bash ./run_benchmark.sh 5


