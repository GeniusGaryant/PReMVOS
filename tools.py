'''
@Author: Gao Yang
@Date: 2020-02-04
@Description: 
'''
import os
import json
import math
import shutil
import glob
import numpy as np
from collections import OrderedDict


def check_json(file_loc):
    with open(file_loc, "r") as f:
        train_dict = json.load(f)
        for _, videos in train_dict.items():
            for _, obj in videos.items():
                for img_number, bbox in obj.items():
                    if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
                        print("img_number: {}".format(img_number))


def transfer_bbox(bbox, use_x=True):
    size = 511 if use_x else 127
    if len(bbox) == 4:
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    else:
        w, h = bbox
    wc_x = w + 0.5 * (w+h)
    hc_x = h + 0.5 * (w+h)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = float(size) / s_x
    w = w * scale_x
    h = h * scale_x
    bbox = [(size-w)//2, (size-h)//2, (size+w)//2, (size+h)//2]
    return bbox


def load_json(file_loc, dataset_name):
    output_dir = "output-" + dataset_name + "/intermediate/refined_proposals/"
    with open(file_loc, "r") as f:
        train_dict = json.load(f)
        for key, videos in train_dict.items():
            for ob_id, obj in videos.items():
                for img_number, bbox in obj.items():
                    assert len(bbox) == 4
                    bbox = transfer_bbox(bbox, use_x=True)
                    bbox[2] -= bbox[0] # w
                    bbox[3] -= bbox[1] # h
                    results_json = [{
                        "bbox": list(map(lambda x: float(round(x, 1)), bbox)),
                    }]
                    folder = os.path.exists(output_dir + key)
                    if not folder:
                        os.makedirs(output_dir + key)
                    save_file = output_dir + key + "/" + img_number + "." + ob_id + ".x" + ".json"
                    with open(save_file, 'w') as f:
                        print("Saving: " + save_file)
                        json.dump(results_json, f)
    print("Done.")


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def split_data(json_dir, dataset_name):
    with open(json_dir, "r") as f:
        train_dict = json.load(f)
    seq_list = []
    for key in train_dict.keys():
        seq_list.append(key)
    splited_seq_lists = chunks(seq_list, 9)
    split_dir = "split_" + dataset_name + "/"
    for i in range(9):
        for sub_seq in splited_seq_lists[i]:
            seq_dir = ""
            target_dir = split_dir + str(i) + "/" + sub_seq
            for sub_path in target_dir.split("/")[:-1]:
                seq_dir += sub_path+"/"
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            with open(target_dir + ".json", "w") as f:
                print("Writing {}".format(target_dir + ".json"))
                random_content = 0
                json.dump(random_content, f)
    print("Done.")


def find_invalidSeq():
    all_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # finished_seqs = [0,1,3,4,5,7,8]
    # unfinished_seqs = [2,6]
    # for seq_number in unfinished_seqs:
    for seq_number in all_seqs:
        # for seq_number in finished_seqs:
        count = 0
        for _, _, files in os.walk("split_trackingnet/" + str(seq_number)):
            for _file in files:
                # filename without ".json"
                # print(_file[:-5])
                if not os.path.exists("record_trackingnet/" + _file):
                    count += 1
                    target_dir = "bug_seq_trackingnet/" + str(seq_number)
                    if not os.path.exists(target_dir):
                        os.mkdir(target_dir)
                    with open(target_dir + "/" + _file, "w") as f:
                        print("Writing {}".format(target_dir + "/" + _file))
                        random_content = 0
                        json.dump(random_content, f)

        print("Seq {} has {} invalid seqs in total.".format(seq_number, count))


def copy_bugSeq():
    bug_seq_path = "bug_seq_got10k/"
    bug_seq_img_path = "bug_imgs_got10k/"
    if not os.path.exists(bug_seq_img_path):
        os.mkdir(bug_seq_img_path)
    for _, _, files in os.walk(bug_seq_path):
        # print(files)
        for seq_file in files:
            seq_name = seq_file[:-5]
            origin_img_path = "/home/zjh/PReMVOS/data/training_dataset/got-10k/train/sequences/" + seq_name
            print("Copy: from {} to {}".format(
                origin_img_path, bug_seq_img_path+seq_name))
            shutil.copytree(origin_img_path, bug_seq_img_path + seq_name)

    print("Done.")


def make_subseq():
    """
    为节省时间，将GPU8本来要跑的剩余序列分给空闲的其他卡
    """
    mission_gpus = [0, 1, 3, 4, 5, 7, 8]
    origin_path = "split_trackingnet/8/"
    rest_files = []
    for _, _, files in os.walk(origin_path):
        for _file in files:
            if not os.path.exists("record_trackingnet/" + _file):
                rest_files.append(_file)
    # print(len(rest_files))
    gpu_numbers = len(mission_gpus)
    splited_seq_lists = chunks(rest_files, gpu_numbers)
    for i in range(gpu_numbers):
        for single_seq in splited_seq_lists[i]:
            sub_seq_path = origin_path + single_seq
            target_path = origin_path + str(mission_gpus[i]) + "/"
            # if not os.path.exists(target_path):
            #     os.mkdir(target_path)
            print("Moving {} to {}".format(sub_seq_path, str(mission_gpus[i])))
            shutil.move(sub_seq_path, target_path)

    print("Done.")


def generate_final_json():
    """
    Final dict structure:
    {
        "LASOT": {
            "seq1": {
                "00": {
                    "query1": [.......]
                }
            }
        },
        "GOT10K": {
            ......
        }
    }
    """
    final_dict = {}

    with open("final_lasot.json", "r") as f:
        print("Opening final_lasot.json..")
        dataset_info = json.load(f)

        dataset_name = []
        for i in sorted(dataset_info.keys()):
            dataset_name.append(i)

        lasot_patch_number = 0
        lasot_seq_number = len(dataset_info[dataset_name[0]].keys())
        for seq_keys, seq_values in sorted(dataset_info[dataset_name[0]].items()):
            for trk, frames in sorted(seq_values.items()):
                # frame_names = []
                for key in sorted(frames.keys()):
                    if len(frames[key]) is not 128:
                        print("[] exists in {} {} {}, poping..".format(
                            dataset_name[0], seq_keys, key))
                        frames.pop(key)
                    else:
                        lasot_patch_number += 1
                    # frame_names.append(key)
                # for i in range(len(frame_names)):
                #     if len(frames[frame_names[i]]) is not 128:
                #         print("[] exists in {} {} {}".format(
                #             dataset_name[0], seq_keys, frame_names[i]))
                #     else:
                #         lasot_patch_number += 1
                        # dataset_info[dataset_name[0]][seq_keys]["00"][frame_names[i]] = dataset_info[dataset_name[0]][seq_keys]["00"][frame_names[i-1]]

                for key in sorted(frames.keys()):
                    dataset_info[dataset_name[0]][seq_keys]["00"]['%06d' % int(key)] = dataset_info[dataset_name[0]][seq_keys]["00"].pop(key)


        final_dict["LASOT"] = dataset_info[dataset_name[0]]
        print("LASOT done, seq_number: {}, patch_number: {}".format(
            lasot_seq_number, lasot_patch_number))

    datasets = ["GOT10K", "TRACKINGNET"]
    for dataset in datasets:
        seq_number = 0
        patch_number = 0

        final_dict[dataset] = {}

        record_path = "record_" + dataset.lower()
        files = sorted(glob.glob(record_path + "/*.json"))

        for each_file in files:
            seq_number += 1
            with open(each_file, "r") as f:
                seq_info = json.load(f)
                seq_name = []
                for i in sorted(seq_info.keys()):
                    seq_name.append(i)

                for seq_keys, seq_values in sorted(seq_info.items()):
                    for trk, frames in sorted(seq_values.items()):
                        frame_names = []
                        for key in sorted(frames.keys()):
                            frame_names.append(key)
                        for i in range(len(frame_names)):
                            patch_number += 1
                            if len(frames[frame_names[i]]) is not 128:
                                print("[] exists in {} {} {} {}".format(
                                    dataset, seq_name[0], seq_keys, frame_names[i]))
                                if i >= 1:
                                    print('supplemented by the former frame')
                                    seq_info[seq_keys]["00"][frame_names[i]] = seq_info[seq_keys]["00"][frame_names[i-1]]
                
                for seq_keys, seq_values in sorted(seq_info.items()):
                    for trk, frames in sorted(seq_values.items()):
                        for key in sorted(frames.keys()):
                            seq_info[seq_keys]["00"]['%06d' % int(key)] = seq_info[seq_keys]["00"].pop(key)

                print("Dealing {}'s seq {}".format(dataset, seq_name[0]))
                final_dict[dataset][seq_name[0]] = seq_info[seq_name[0]]

        print("{} seq_numbers: {}, patch_numbers: {}".format(dataset, seq_number, patch_number))

    with open("final_all_2.json", "w") as f:
        json.dump(final_dict, f, sort_keys=True)
        print("Saved the final json.")

    print("Done.")


def deal_keys():
    with open('final_all.json', 'r') as f:
        data = json.load(f)

    new_data = {}
    new_data['LASOT'] = data['lasot']
    new_data['GOT10K'] = data['got10k']
    new_data['TRACKINGNET'] = data['trackingnet']

    json.dump(new_data, open('final_temp.json', 'w'), sort_keys=True)


def draw(image, gt_boxes):
    # image (B, 3, w, h)   tensor
    # gt_boxes (B, 4)      tensor

    batch_size = image.size(1)
    for i in range(batch_size):
        image = image[i].permute(1, 2, 0).detach().cpu().numpy()
        image = cv2.rectangle(image, tuple([int(m) for m in gt_boxes[i][0:2].data]),
                                tuple([int(m) for m in gt_boxes[i][2:4].data]), (0, 0, 255), 2)
        cv2.imwrite('debug/test{}.jpg'.format(i), image)


if __name__ == "__main__":

    # json_dir = "data/training_dataset/trackingnet/train.json"
    # json_dir = "data/training_dataset/got-10k/train.json"
    # json_dir = "data/training_dataset/lasot/train.json"

    print("In main:")

    json_dir = "data/training_dataset/coco/train2017.json"
    # load_json(json_dir, "coco")
    # print("coco done.")

    # json_dir = "data/training_dataset/det/train.json"
    # load_json(json_dir, "det")
    # print("det done.")

    # json_dir = "data/training_dataset/vid/train.json"
    # load_json(json_dir, "vid")
    # print("vid done.")

    # json_dir = "data/training_dataset/yt_bb/train.json"
    # load_json(json_dir, "yt_bb")
    # print("yt_bb done.")

    # json_dir = "data/training_dataset/coco/train2017.json"
    # load_json(json_dir, "lasot")
    # print("lasot done.")

    split_data(json_dir, "coco")


    # check_json(json_dir)
