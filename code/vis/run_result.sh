#!/usr/bin/env bash

#####################清理上次结果#######################
if [ -d ./exp_images ]; then
  rm -rf ./exp_images
fi
if [ -d ./num_res ]; then
  rm -rf ./num_res
fi
if [ -d ./report ]; then
  rm -rf ./report
fi
if [ -d ./datasets ]; then
  rm -rf ./datasets
fi
mkdir ./exp_images ./num_res ./report ./datasets
echo "CREATED FOLDERS !"
#####################开始工作#########################
python ./get_best.py
echo "COPIED BEST RUNTIME !"
python ./plot_picture.py
echo "FINISHED PLOTTING !"
rm -rf ./datasets/hangseng ./datasets/dax ./datasets/ftse ./datasets/sp ./datasets/nikkei
echo "CLEANED PLOTTING CACHED FILES !"
#####################计算指标#########################
python compute_metric.py >> ./report/refpoint.txt
echo "FINISHED　COMPUTING METRICS !"
#####################统计结果#########################
# python compute_statistics.py 1 >> ./report/hangseng.txt
# python compute_statistics.py 2 >> ./report/dax.txt
# python compute_statistics.py 3 >> ./report/ftse.txt
python compute_statistics.py 4 >> ./report/sp.txt
# python compute_statistics.py 5 >> ./report/nikkei.txt
echo "FINISHED　STATISTICAL PROCESSING !"
#####################清理缓存#########################
#if [ -d ./__pycache__ ]; then
#  rm -rf ./__pycache__
#fi
echo "FINISHED COMPILING CACHED FILES !"