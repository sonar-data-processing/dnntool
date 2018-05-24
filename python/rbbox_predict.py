#!/usr/bin/python
import os
import time
import argparse
import fnmatch
import numpy as np
import keras
import cv2
import drawing
import rbbox
import keras.backend as KB
import keras.layers as KL
from keras.models import Model
from keras.preprocessing import image
from annotation_utils import *

parser = argparse.ArgumentParser(
    description="Test rotated bounding box annotations")

parser.add_argument(
    "image_folder",
    help="Folder containg the image dataset")

def _main_(args):
    target_size = (224, 224)

    model_rbbox_regr = rbbox.get_model_rbbox_regressor(target_size)
    model_rbbox_regr.load_weights("/home/gustavoneves/sources/dnntool/weights/rbbox-gemini-final.h5")
    model_rbbox_regr.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    for root, dirs, files in os.walk(args.image_folder):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        files = fnmatch.filter(files, '[!flip]*')
        for name in files:
            im_path = os.path.join(root, name)
            im = cv2.imread(im_path)
            class_id, gt = get_rbbox_annotation(im_path)

            box = gt[:4]
            rbox_gt = gt[4:]

            rbox = rbbox.predict(im, class_id, box, model_rbbox_regr, target_size=target_size, use_bbox=True)
            rbox_gt[0:2] += box[0:2]
            rbox_gt = rbox2vert(rbox_gt)
            pts_gt = vert2points(rbox_gt)

            rbox = rbox2vert(rbox)
            pts = vert2points(rbox)

            drawing.points(im, pts_gt)
            drawing.points(im, pts, colors=(0, 0, 255))

            cv2.rectangle(im,(box[0], box[1]),(box[2], box[3]),(0,255,0),3)
            cv2.imshow("annotation", im)

            if cv2.waitKey(15) & 0xFF == ord('q'):
                exit()

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)
