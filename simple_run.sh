#!/bin/bash

# DATASET=coco
# GPU_NUMBER=0
# IMG_DIR=/home/zjh/PReMVOS/split_$DATASET/$GPU_NUMBER
# IMG_DIR_LEN=${#IMG_DIR}

# function lstdir() {
#     for file in `ls $1`
#     do
#         local path=$1"/"$file
#         if [ -d $path ]; then
#             lstdir $path;
#         elif [ -f $path ]; then
#             # cut suffix
#             path=${path%.*}
#             path=${path:$IMG_DIR_LEN+1}
#             echo "################# GENERATING $path PROPOSALS #################"
#             cd code
#             ReID_CONFIG=./ReID_net/configs/run
#             echo "$ReID_CONFIG"
#             CUDA_VISIBLE_DEVICES=$GPU_NUMBER ./ReID_net/main.py "$ReID_CONFIG" $path
#             cd ..
#         fi
#     done
# }
# lstdir $IMG_DIR;


# if [ ! -d "$ReID_PROP_LOC" ]; then
#   echo "################# GENERATING ReID PROPOSALS #################"
#   cd code
#   ReID_CONFIG=./ReID_net/configs/run
#   echo "$ReID_CONFIG"
#   CUDA_VISIBLE_DEVICES=1 ./ReID_net/main.py "$ReID_CONFIG"
#   cd ..
# fi





# debug part
path=train2017/000000000009
echo "################# GENERATING $path PROPOSALS #################"
cd code
ReID_CONFIG=./ReID_net/configs/run
echo "$ReID_CONFIG"
CUDA_VISIBLE_DEVICES=$GPU_NUMBER ./ReID_net/main.py "$ReID_CONFIG" $path
cd ..