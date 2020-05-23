import glob
from scipy.misc import imread

import numpy as np

import Measures as Measures
import Util as Util
import datasets.Util.Util as dataUtil

SHARPMASK_PATH = '/home/' + dataUtil.username() + '/vision/from_paul/deepmask_VOT/data/out_sharpmask/'
VOC_PATH = "/work/" + dataUtil.username() + "/data/PascalVOC/benchmark_RELEASE/dataset/"
VOC_SEGM_PATH = "/home/" + dataUtil.username() + "/vision/savitar/forwarded/pascalVOC_instance1/valid/JPEGImages/"
THRESH=0.6

#Calculate the IOU for masks generated by sharpmask.
def eval_sharpmask():
  data = {}
  n_imgs = 0
  ious = []

  with open(SHARPMASK_PATH + "list.txt") as f:
    lines = f.readlines()

    for line in lines:
      id, file, x1, y1, x2, y2 = line.split()
      bbox = [x1,y1,x2,y2]

      filename = file.split('/')[-1]

      if filename in list(data.keys()):
        if bbox in data[filename]:
          continue
        data[filename].append(bbox)
      else:
        data[filename] = [bbox]

      pred_file = SHARPMASK_PATH + "masks/" + id + ".png"
      pred = imread(pred_file)
      pred = pred / 255.0
      measures = get_measures(pred, filename)
      ious.append(measures['iou'])
      n_imgs+=1

  avg_iou = sum(ious) / n_imgs
  print("n_imgs: " + repr(n_imgs) + "\n" + "IOU_sharpmask: " + repr(avg_iou))


def eval_ours():
  files = glob.glob(VOC_SEGM_PATH + "*.png")
  n_imgs = 0
  ious = []

  for file in files:
    pred = imread(file)
    instances = np.unique(pred)
    instances = np.delete(instances, 0)

    if len(instances) == 0:
      print("File " + file + " does not have any instances.")

    for  instance in instances:
      pred_inst = np.zeros_like(pred)
      pred_inst[np.where(pred == instance)] = 1

      measures = get_measures(pred_inst, file.rsplit('_',1)[0].split('/')[-1])
      ious.append(measures['iou'])
      n_imgs+=1

  avg_iou = sum(ious) / n_imgs
  print("n_imgs: " + repr(n_imgs) + "\n" + "IOU_ours: " + repr(avg_iou))
    

def get_measures(pred, filename):
  # pred = imread(pred_file)
  pred = np.where(pred > THRESH, 1, pred)
  # pred = np.max(pred, axis=2)

  filename = filename.replace(".jpg", '')
  filename = filename + ".png"
  gt_file = VOC_PATH + "SegmentationObject/" + filename
  gt = imread(gt_file)

  target = Util.get_best_overlap(pred_mask=pred, gt=gt,
                                 ignore_classes=[0, 255])
  measures = Measures.compute_measures_for_binary_segmentation( pred, target)
  
  return measures
if __name__ == '__main__':
    # eval_sharpmask()
    eval_ours()
