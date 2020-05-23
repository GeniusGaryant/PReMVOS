#!/usr/bin/env python
import sys
import os
import time
import json

from Engine import Engine
from Config import Config
from Log import log
import tensorflow as tf


def init_log(config):
    log_dir = config.dir("log_dir", "logs")
    model = config.str("model")
    filename = log_dir + model + ".log"
    verbosity = config.int("log_verbosity", 3)
    log.initialize([filename], [verbosity], [])


def main(_):
    assert len(sys.argv) == 3, "usage: main.py <config>"
    config_path = sys.argv[1]
    seq_name = sys.argv[2]
    assert os.path.exists(config_path), config_path

    try:
        config = Config(config_path)
    except ValueError as e:
        print("Malformed config file:", e)
        return -1
    # init_log(config)
    config.initialize()

    dataset_name = "coco"
    # for coco:  seq_name: 'train2017/000000287469'
    # for det:   seq_name: 'a/n02028035_10675'
    # for vid:   seq_name: 'd/ILSVRC2015_train_01085000'
    # for yt_bb: seq_name: 'train0021/0/CbO2dReBn1o'

    seq_dir = ""
    for sub_path in seq_name.split("/")[:-1]:
        seq_dir += sub_path+"/"
    save_dir = "../score_" + dataset_name + "/" + seq_dir
    if not os.path.exists(save_dir + seq_name.split("/")[-1] + ".json"):
        start = time.time()
        # print("Running seq_file: {}".format(seq_name))
        engine = Engine(config, seq_name)
        engine.run()
        print("seq_file {} finished, took {} sec\n\n\n".format(seq_name, time.time()-start))
    else:
        print("The seq_file {} already exists, move to the next..".format(seq_name))


    # if dataset_name == "coco":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/coco/train2017.json"
    # elif dataset_name == "det":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/det/train.json"
    # elif dataset_name == "vid":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/vid/train.json"
    # elif dataset_name == "yt_bb":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/yt_bb/train.json"
    # elif dataset_name == "lasot":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/lasot/train.json"
    # elif dataset_name == "got-10k":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/got-10k/train.json"
    # elif dataset_name == "trackingnet":
    #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/trackingnet/train.json"
    # # elif dataset_name == "visdrone":
    # #     anno_name = "/home/zjh/PReMVOS/data/training_dataset/visdrone/train.json"
    # else:
    #     raise AssertionError

    # f = open(anno_name, "r")
    # dataset_dict = json.load(f)
    # for seq_name in dataset_dict.keys():
    #     # for coco:  seq_name: 'train2017/000000287469'
    #     # for det:   seq_name: 'a/n02028035_10675'
    #     # for vid:   seq_name: 'd/ILSVRC2015_train_01085000'
    #     # for yt_bb: seq_name: 'train0021/0/CbO2dReBn1o'

    #     seq_dir = ""
    #     for sub_path in seq_name.split("/")[:-1]:
    #         seq_dir += sub_path+"/"
    #     save_dir = "/home/zjh/PReMVOS/score_" + dataset_name + "/" + seq_dir
    #     if not os.path.exists(save_dir + seq_name.split("/")[-1] + ".json"):
    #         start = time.time()
    #         print("Running seq_file: {}".format(seq_name))
    #         engine = Engine(config, seq_name)
    #         engine.run()
    #         print("seq_file {} finished, took {} sec".format(seq_name, time.time()-start))
    #     else:
    #         print("The seq_file {} already exists, move to the next..".format(seq_name))
    # f.close()


if __name__ == '__main__':
    # for profiling. Note however that this will not be useful for the execution of the tensorflow graph,
    # only for stuff like initialization including creation of the graph, loading of weights, etc.
    # import cProfile
    # cProfile.run("tf.app.run(main)", sort="tottime")
    tf.app.run(main)
