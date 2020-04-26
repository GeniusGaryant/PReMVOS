import random
from annoy import AnnoyIndex
import json
from tqdm import tqdm
from concurrent import futures
import time
import sys
import copy
import os.path as osp
import os
import numpy as np

# 1 store tree
# 2 load tree and search nn
# 3 do above both
mode = 4
json_file = 'final_all_2.json'
idx_stru = AnnoyIndex(128, 'angular')
nn_num = 64

with open(json_file, 'r') as f:
    data = json.load(f)

print("documenting data statistics...")

i = 0
key2id = copy.deepcopy(data)
id2key = []
for dts, videos in data.items():
    for vid, tracks in videos.items():
        for trk, frames in tracks.items():
            for frm, em_vec in frames.items():
                if frm.isdigit() and em_vec is not None:
                    key2id[dts][vid][trk][frm] = i
                    id2key.append("/".join([dts, vid, trk, frm]))
                    if mode == 1 or mode == 3:
                      idx_stru.add_item(i, em_vec)
                    i += 1
                else:
                    print(dts, vid, trk, frm, em_vec)

if mode == 1 or mode == 3:
    idx_stru.build(100)
    idx_stru.save(json_file.split('.')[0])
elif mode == 2:
    idx_stru.load(json_file.split('.')[0])
    print('loading index structure done...')

id2key_file = osp.join("{}_id2key.npy".format(json_file.split(".")[0]))
if not osp.exists(id2key_file): np.save(id2key_file, np.array(id2key))

if mode ==2 or mode == 3:
    print('constructing nn dict...')

    def do_video(dts, vid, tracks):
        
        save_path = osp.join("{}_nn".format(json_file.split('.')[0]), dts)
        if not osp.exists(save_path): os.makedirs(save_path)
        nn_file = osp.join(save_path, vid)
        if osp.exists(nn_file): return

        tracks_data = {}
        rtv_num = 50000 if dts == 'LASOT' else 10000
        for trk, frames in tracks.items():
            tracks_data[trk] = {}
            for frm in frames.keys():
                tag_trk_name = '/'.join([vid, trk])
                rtv_idx = idx_stru.get_nns_by_item(key2id[dts][vid][trk][frm], rtv_num)
                random.shuffle(rtv_idx)
                
                rtv_trk_names = []
                rtv_patch_names = []
                for j in rtv_idx:
                    rtv_trk_name = '/'.join(id2key[j].split('/')[1:3])
                    if rtv_trk_name != tag_trk_name and rtv_trk_name not in rtv_trk_names:
                        rtv_trk_names.append(rtv_trk_name)
                        # rtv_patch_names.append(id2key[j])
                        rtv_patch_names.append(j)
                    if len(rtv_patch_names) >= nn_num:
                        break
                if len(rtv_patch_names) < nn_num:
                    print('random sampling...')
                    # rtv_patch_names += random.sample(id2key, nn_num - len(rtv_patch_names))
                    rtv_patch_names += random.sample(range(len(id2key)), nn_num - len(rtv_patch_names))

                tracks_data[trk][frm] = rtv_patch_names

        json.dump(tracks_data, open(nn_file, 'w'), sort_keys=True)

        #return dts, vid, tracks_data

    def excute(num_threads=24):
        nn_data = {}
        for dts in data.keys():
            nn_data[dts] = {}

        with futures.ProcessPoolExecutor(max_workers=num_threads) as excutor:
            fs = [excutor.submit(do_video, dts, vid, tracks) for dts, videos in data.items() for vid, tracks in videos.items()]

    since = time.time()
    excute()
   
    nn_path = "{}_nn".format(json_file.split(".")[0])
    nn_data = {}
    i = 0
    for dts in sorted(os.listdir(nn_path)):
        nn_data[dts] = {}
        for vid in sorted(os.listdir(osp.join(nn_path, dts))):
            with open(osp.join(nn_path, dts, vid), 'r') as f:
                trk_data = json.load(f)
            nn_data[dts][vid] = trk_data
            i += 1
            print("{} | {}.{} completed".format(i, dts, vid))
    json.dump(nn_data, open("{}.json".format(nn_path), 'w'), sort_keys=True)

    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
