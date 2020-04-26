#!/bin/bash

# Add ReID to proposals
DATASET=lasot
GPU_NUMBER=1
IMG_DIR=/home/zjh/PReMVOS/split_$DATASET/$GPU_NUMBER
files=$(ls $IMG_DIR | cut -d . -f1)
for i in $files
do
    echo "################# GENERATING $i PROPOSALS #################"
    cd code
    ReID_CONFIG=./ReID_net/configs/run
    echo "$ReID_CONFIG"
    CUDA_VISIBLE_DEVICES=$GPU_NUMBER ./ReID_net/main.py "$ReID_CONFIG" $i
    cd ..
done



# # Problem seq
# # seq="GOT-10k_Train_008628"
# seq="GOT-10k_Train_008630"
# # seq="GOT-10k_Train_009058"
# echo "################# DEBUG $seq #################"
# cd code
# ReID_CONFIG=./ReID_net/configs/run
# # echo "$ReID_CONFIG"
# CUDA_VISIBLE_DEVICES=$GPU_NUMBER ./ReID_net/main.py "$ReID_CONFIG" $seq
# cd ..



# IMG_DIR=/home/zjh/PReMVOS/split_trackingnet/1
# dir=$(ls -l $IMG_DIR |awk '/^d/ {print $NF}')
# for i in $dir
# do
#   echo $i
#   # echo "################# GENERATING ReID PROPOSALS #################"
#   # cd code
#   # ReID_CONFIG=./ReID_net/configs/run
#   # echo "$ReID_CONFIG"
#   # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ./ReID_net/main.py "$ReID_CONFIG" $i
#   # cd ..
# done



# if [ ! -d "$ReID_PROP_LOC" ]; then
#   echo "################# GENERATING ReID PROPOSALS #################"
#   cd code
#   ReID_CONFIG=./ReID_net/configs/run
#   echo "$ReID_CONFIG"
#   CUDA_VISIBLE_DEVICES=1 ./ReID_net/main.py "$ReID_CONFIG"
#   cd ..
# fi

