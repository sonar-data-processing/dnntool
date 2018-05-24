#!/usr/bin/python
import os
import math
import argparse
import fnmatch
import numpy as np
import drawing
import cv2
import rbbox
import annotation_utils
import json
from pycocotools.coco import COCO
from datetime import datetime

parser = argparse.ArgumentParser(
    description="Generate results")

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
        return "{}-result-resnet50".format(detector)

def _get_ground_truth(img_path):
    gt_id, gt = annotation_utils.get_rbbox_annotation(img_path)
    gt_box = gt[:4]
    gt_rbox = gt[4:]
    gt_rbox[0:2] += gt_box[0:2]
    return gt_id, gt_box, gt_rbox

def _create_rbbox_mask(size, rbbox):
    mask = np.zeros(size, dtype=np.uint8)
    points = annotation_utils.rbox2points(rbbox)
    points =  points.astype(int)
    cv2.fillConvexPoly(mask, points, 255)
    return mask

def _main_(args):
    coco=COCO('coco_annotations.json')
    
    cats = coco.loadCats(coco.getCatIds())
    print cats

    img_ids = coco.getImgIds()
    img_ids = sorted(img_ids)
    imgs = coco.loadImgs(img_ids)

    suffix = _get_suffix(args.detector)
    
    coco_results = []

    for img in imgs:
        img_path = img['coco_url']
        print img_path
        I = cv2.imread(img_path)
        h, w, _ = I.shape
        
        gt_id, gt_box, gt_rbox = _get_ground_truth(img_path)

        result_path = rbbox.get_result_filename(img_path, suffix='-'+suffix)
        ids, boxes, rboxes, scores = rbbox.load_result(result_path)

        # labels = [yolo_labels[id] for id in ids]
        labels = 'jequitaia'
        cat_ids = coco.getCatIds(catNms = labels)

        drawing.boxes(I, boxes, labels, scores)
        drawing.rboxes(I, rboxes)
        drawing.rbox(I, gt_rbox, color=(0, 0, 255))
        item = dict()
        if rboxes:
            if cat_ids:
                for i, rbox in enumerate(rboxes):
                    b = boxes[i].astype(int)
                    item['image_id'] = img['id']
                    # item['category_id'] = cat_ids[i]
                    item['category_id'] = 1
                    item['bbox'] = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
                    item['score'] = scores[i]
                    coco_results.append(item)
                    mask_r = _create_rbbox_mask((h, w), rbox)
                    cv2.imshow("mask", mask_r)
            
        cv2.imshow("result", I)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    json.dump(coco_results, open('coco_results.json', 'w'))

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

