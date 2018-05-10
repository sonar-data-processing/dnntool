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
    description="Load YOLO Annotations")

parser.add_argument(
    "image_folder",
    help="Folder containg the image dataset")


if __name__ == '__main__':
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.image_folder):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        for name in files:
            im_path = os.path.join(root, name)

            _, box = load_yolo_annotation(im_path)
            _, rbox = load_yolo_rot_annotation(im_path)

            im = cv2.imread(im_path)

            h = im.shape[0]
            w = im.shape[1]

            box = rescale_box((w, h), box)
            box = box2vert(box)

            rbox = rbox2vert(rbox, (w, h))
            pts = vert2points(rbox)

            cv2.rectangle(im,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)
            drawing.points(im, pts)
            cv2.imshow("annotation", im);

            if cv2.waitKey(200) & 0xFF == ord('q'):
                exit()
