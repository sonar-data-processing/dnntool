#!/usr/bin/python
import numpy as np
import drawing
import cv2
import csv
import argparse
from annotation_utils import *

parser = argparse.ArgumentParser(
    description="Load faster-rcnn annotations")

parser.add_argument(
    "annotations",
    help="Text file with annotations")

parser.add_argument(
    '--type',
    default='box',
    choices=list(['box', 'rbox']),
    help="Annotation type")

def _main_():
    args = parser.parse_args()
    anns_filename = args.annotations
    anns_type = args.type

    with open(anns_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            im = cv2.imread(row[0])
            box = np.array(row[1:5], dtype=np.float32)
            box = box.astype(int)
            cv2.rectangle(im,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)

            if anns_type=='rbox':
                rbox_gt = np.array(row[5:10], dtype=np.float32)
                print rbox_gt
                rbox_gt[0:2] += box[0:2]
                rbox_gt = rbox2vert(rbox_gt)
                pts_gt = vert2points(rbox_gt)
                drawing.points(im, pts_gt)

            cv2.imshow('image', im)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                exit()

if __name__ == '__main__':
    _main_()

