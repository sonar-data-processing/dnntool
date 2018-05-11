#!/usr/bin/python
import os
import math
import argparse
import fnmatch
import numpy as np
import drawing
import cv2
import cv2.cv as cv
from annotation_utils import *

parser = argparse.ArgumentParser(
    description="Test rotated bounding box annotations")

parser.add_argument(
    "image_folder",
    help="Folder containg the image dataset")


def _get_rbbox_annotation(filepath, suffix=".txt"):
    with open(get_annotation_path(filepath, suffix), 'r') as txtfile:
        id, x1, y1, x2, y2, cx, cy, w, h, t = txtfile.readline().rstrip().split(' ')
        return int(id), np.array([x1, y1, x2, y2, cx, cy, w, h, t], dtype=np.float32)

if __name__ == '__main__':
    args = parser.parse_args()
    for root, dirs, files in os.walk(args.image_folder):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        for name in files:
            im_path = os.path.join(root, name)
            im = cv2.imread(im_path)
            _, gt = _get_rbbox_annotation(im_path)
            box = gt[:5]
            rbox = gt[4:]
            rbox[0:2] += box[0:2]
            rbox = rbox2vert(rbox)
            pts = vert2points(rbox)
            cv2.rectangle(im,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)
            drawing.points(im, pts)
            cv2.imshow("annotation", im);
            if cv2.waitKey(200) & 0xFF == ord('q'):
                exit()
