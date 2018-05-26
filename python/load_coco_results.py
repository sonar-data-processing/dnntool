import sys
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(
    description="Load COCO annotations format")

parser.add_argument(
    "annotation_filepath",
    help="Folder containg the image dataset")

parser.add_argument(
    "result_filepath",
    help="Folder containg the image dataset")

args = parser.parse_args()

annType = ['segm','bbox','keypoints']
annType = annType[1]

cocoGt=COCO(args.annotation_filepath)
cocoDt=cocoGt.loadRes(args.result_filepath)

imgIds=sorted(cocoGt.getImgIds())

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.params.areaRngLbl = ['all']
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()