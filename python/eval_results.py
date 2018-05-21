#!/usr/bin/python
import os
import math
import argparse
import fnmatch
import numpy as np
import drawing
import cv2
import cv2.cv as cv
import rbbox
import annotation_utils

parser = argparse.ArgumentParser(
    description="Evaluate the detection results")

parser.add_argument(
    "image_folder",
    help="Folder containg the image dataset")

parser.add_argument(
    '--detector',
    default='yolo',
    choices=list(['yolo', 'faster_rcnn']),
    help="Detector")


yolo_labels = ["ssiv_bahia", "jequitaia", "balsa"]

def _get_suffix(detector):
        return "-{}-result-resnet50".format(detector)

def _get_ground_truth(img_path):
    gt_id, gt = annotation_utils.get_rbbox_annotation(img_path)
    gt_box = gt[:4]
    gt_rbox = gt[4:]
    gt_rbox[0:2] += gt_box[0:2]
    return gt_id, gt_box, gt_rbox


def _main_(args):
    suffix = _get_suffix(args.detector)
    for root, dirs, files in os.walk(args.image_folder):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)

        for name in files:
            img_path = os.path.join(root, name)
            img = cv2.imread(img_path)

            gt_id, gt_box, gt_rbox = _get_ground_truth(img_path)

            result_path = rbbox.get_result_filename(img_path, suffix=suffix)
            ids, boxes, rboxes, scores = rbbox.load_result(result_path)

            labels = [yolo_labels[id] for id in ids]
           
            drawing.boxes(img, boxes, labels, scores)
            drawing.rboxes(img, rboxes)
            drawing.rbox(img, gt_rbox, color=(0, 0, 255))

            cv2.imshow("result", img)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                exit()

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

