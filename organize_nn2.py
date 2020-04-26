import json
import os
import os.path as osp
json_file = 'final_all_2.json'

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
