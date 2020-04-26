from .Similarity import SimilarityDataset
import glob
import json
from pycocotools.mask import toBbox
import cv2
import os


class DAVISForwardSimilarityDataset(SimilarityDataset):
    def __init__(self, config, subset, coord, seq_path):
        old_proposal_directory = config.str("bb_input_dir", None)
        data_directory = config.str("image_input_dir", None)

        annotations = []
        name = seq_path
        files = sorted(
            glob.glob(old_proposal_directory + name + "/*.json"))
        for file in files:
            timestep = file.split('/')[-1].split('.json')[0]
            # timestep = str(int(timestep)) # 去掉trackingnet序列中前面的0,其他数据集不需要
            with open(file, "r") as f:
                proposals = json.load(f)
            for prop_id, proposal in enumerate(proposals):
                img_file = data_directory + name + "/" + timestep + ".jpg"
                if not os.path.exists(img_file):
                    AssertionError("img_file: {} lacked.".format(img_file))
                catagory_id = 1
                tag = name + '/' + timestep + '___' + str(prop_id)
                bbox = proposal["bbox"]
                ann = {"img_file": img_file,
                        "category_id": catagory_id, "bbox": bbox, "tag": tag}

                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                annotations.append(ann)

        super(DAVISForwardSimilarityDataset, self).__init__(
            config, subset, coord, annotations, n_train_ids=1)
