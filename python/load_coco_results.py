import sys
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annType = ['segm','bbox','keypoints']
annType = annType[1]

cocoGt=COCO('coco_annotations.json')
cocoDt=cocoGt.loadRes('coco_results_faster_rcnn.json')

# dts = cocoDt.loadAnns(cocoDt.getAnnIds(cocoGt.getImgIds()))
# print dts

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()