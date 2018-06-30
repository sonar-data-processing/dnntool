import sys
import skimage.io as io
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import drawing
import annotation_utils
from pycocotools.coco import COCO

from threading import Thread
import curses

key_pressed = False
parser = argparse.ArgumentParser(
    description="Load COCO results format")


parser.add_argument(
    "annotation_filepath",
    help="Folder containg the image dataset")

args = parser.parse_args()

coco=COCO(args.annotation_filepath)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['balsa'])
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

plt.ion()
plt.show()

im = None
for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]

    # load and display image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    
    I = io.imread(img['coco_url'])

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for ann in anns:
        seg = ann['segmentation']
        box = ann['bbox']
        cv2.rectangle(I,(box[0], box[1]),(box[0]+box[2], box[1]+box[3]),(0,255,0),3)
        drawing.points(I, annotation_utils.vert2points(seg[0]))
    
    cv2.imshow("image", I)

    if cv2.waitKey(15) & 0xFF == ord('q'):
        exit()