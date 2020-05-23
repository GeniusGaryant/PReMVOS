import numpy as np
import os
import time
from .Forwarder import Forwarder
from Log import log
import glob
import json
from collections import OrderedDict


class ReIDForwarder(Forwarder):

    def __init__(self, engine):
        super(ReIDForwarder, self).__init__(engine)
        self.network = None
        self.data = None

    def get_all_latents_from_network(self, y_softmax):
        tag = self.network.tags
        n_total = self.data.num_examples_per_epoch()
        n_processed = 0
        ys = []
        tags = []
        start_all = time.time()
        while n_processed < n_total:
            start = time.time()
            y_val, tag_val = self.session.run([y_softmax, tag])
            y_val = y_val[0]
            curr_size = y_val.shape[0]
            for i in range(curr_size):
                ys.append(y_val[i])
                tags.append(tag_val[i].decode('utf-8'))
            n_processed += curr_size
            print(n_processed, "/", n_total, " elapsed: ", time.time() - start)
        print("Process finished, took {} sec".format(time.time() - start_all))

        self.export_data(ys, tags)

    def export_data(self, ys, tags):
        print("EXPORTING DATA")

        old_proposal_directory = self.config.str("bb_input_dir", None)
        # new_proposal_dir = self.config.str("output_dir", None)

        all_proposals = {}
        # Read in all proposals
        folders = sorted(glob.glob(old_proposal_directory + '*/'))
        for folder in folders:
            seq = folder.split('/')[-2]
            name = seq
            files = sorted(
                glob.glob(old_proposal_directory + name + "/*.json"))

            all_proposals[name] = {}
            all_proposals[name]["00"] = {}

            for file in files:
                timestep = file.split('/')[-1].split('.json')[0]
                # Get proposals:
                with open(file, "r") as f:
                    # [{'bbox': [945.0, 293.0, 52.0, 28.0]}]
                    proposals = json.load(f)
                    the_bbox = proposals[0]["bbox"]
                    print("the_bbox: {}".format(the_bbox))
                    if the_bbox == [0.0, 0.0, 0.0, 0.0]:
                        print("________timestep: {}".format(timestep))
                
                all_proposals[name]["00"][timestep] = []

                # all_proposals[name+'/'+timestep] = proposals

        print("READ IN ALL PROPOSALS")

        # Insert embeddings into proposals
        for y, tag in zip(ys, tags):
            nametime, prop_id = tag.split('___')
            name, timestep = nametime.split("/")
            # prop_id = int(prop_id)
            y = np.array(y).tolist()
            all_proposals[name]["00"][timestep] = y

        print("INSERTED EMBEDDINGS")

        # with open("/home/zjh/PReMVOS/final_lasot_2.json", "w") as f:
        #     final = {"LASOT": all_proposals}
        #     # final = all_proposals
        #     json.dump(final, f)
        
        print("SAVED ALL PROPOSALS WITH REID VECTOR")

        # # Save out to file
        # # for set_id, set in enumerate(sets):
        # folders = sorted(glob.glob(old_proposal_directory + '*/'))
        # for folder in folders:
        #     seq = folder.split('/')[-2]
        #     name = seq
        #     files = sorted(
        #         glob.glob(old_proposal_directory + name + "/*.json"))
        #     for file in files:
        #         timestep = file.split('/')[-1].split('.json')[0]

        #         # Save new proposals:
        #         new_file = new_proposal_dir + name + "/" + timestep + ".json"
        #         if not os.path.exists(new_proposal_dir + name):
        #             os.makedirs(new_proposal_dir + name)
        #         with open(new_file, 'w') as f:
        #             json.dump(all_proposals[name+'/'+timestep], f)

    """
    Final dict structure:
    {
        "visdrone": {
            "seq1": {
                "00": {
                    "query1": [.......]
                }
            }
        },
        "got-10k": {
            ......
        }
    }
    """
    def get_latents_and_export_data(self, y_softmax, seq_path):
        print("EXPORTING DATA")

        old_proposal_directory = self.config.str("bb_input_dir", None)
        all_proposals = {}

        name = seq_path
        all_proposals[name] = {}
        files = sorted(
            glob.glob(old_proposal_directory + seq_path + "/*.json"))
        for file in files:
            file_name = file.split('/')[-1].split('.json')[0]
            ob_id = file_name.split('.')[1]
            timestep = file_name.split('.')[0]
            # timestep = str(int(timestep)) # 去掉trackingnet序列中前面的0
            all_proposals[name][ob_id] = {}
            # print("all_proposals[{}][{}][{}]: initialized".format(name, ob_id, timestep))

        tag = self.network.tags
        n_total = self.data.num_examples_per_epoch()
        n_processed = 0
        start_all = time.time()

        # problem_frame = []
        while n_processed < n_total:
            start = time.time()
            try:
                y_val, tag_val = self.session.run([y_softmax, tag])
                y_val = y_val[0]
                curr_size = y_val.shape[0]
                for i in range(curr_size):
                    nametime, _ = tag_val[i].decode('utf-8').split("___")
                    file_name = nametime.split("/")[-1]
                    ob_id = file_name.split('.')[1]
                    timestep = file_name.split('.')[0]
                    y = np.array(y_val[i]).tolist()
                    all_proposals[name][ob_id][timestep] = y
                n_processed += curr_size
            except:
                print("Problem: {}".format(file_name))
                all_proposals[name][ob_id][timestep] = [0.] * 128
                n_processed += 1
            print(n_processed, "/", n_total, " elapsed: ", time.time() - start)

        print("Process finished, took {} sec".format(time.time() - start_all))

        dataset_name = "coco"
        seq_dir = ""
        for sub_path in seq_path.split("/")[:-1]:
            seq_dir += sub_path+"/"
        save_dir = "../score_" + dataset_name + "/" + seq_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + seq_path.split("/")[-1] + ".json", "w") as f:
            final = all_proposals
            json.dump(final, f)
            print("SAVED ALL PROPOSALS WITH REID VECTOR")


    def forward(self, network, data, seq_path, save_results=True, save_logits=False):
        self.network = network
        self.data = data
        output = self.network.get_output_layer().outputs
        self.get_latents_and_export_data(output, seq_path)
