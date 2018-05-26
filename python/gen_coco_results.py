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
import pycocotools
from pycocotools.mask import encode, decode
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(
    description="Generate COCO result formats")

parser.add_argument(
    "annotation_filepath",
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
    coco=COCO(args.annotation_filepath)

    # catIds = coco.getCatIds(catNms=['ssiv_bahia'])
    # catIds = coco.getCatIds()
    # imgIds = coco.getImgIds(catIds=catIds)
    imgIds = coco.getImgIds()
    imgIds = sorted(imgIds)
    imgs = coco.loadImgs(imgIds)

    suffix = _get_suffix(args.detector)
    
    coco_results = []

    cnt = 0
    for img in imgs:
        img_path = img['coco_url']
        I = cv2.imread(img_path)
        h, w, _ = I.shape

        gt_id, gt_box, gt_rbox = _get_ground_truth(img_path)

        result_path = rbbox.get_result_filename(img_path, suffix='-'+suffix)
        ids, boxes, rboxes, scores = rbbox.load_result(result_path)

        labels = [yolo_labels[id] for id in ids]

        drawing.boxes(I, boxes, labels, scores)
        drawing.rboxes(I, rboxes)
        drawing.rbox(I, gt_rbox, color=(0, 0, 255))
        item = dict()
        if rboxes:
            for i, rbox in enumerate(rboxes):
                cat_id = coco.getCatIds(catNms = [labels[i]])[0]
                b = boxes[i].astype(int)
                item['image_id'] = img['id']
                item['category_id'] = cat_id
                item['bbox'] = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
                item['score'] = scores[i]
                coco_results.append(item)
                bimask = _create_rbbox_mask((h, w), rbox)
                rle = encode(np.asfortranarray(bimask))
                item['segmentation'] = rle
            
        cv2.imshow("result", I)
        if (cnt==0):
            cv2.waitKey()
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        cnt+=1

    json.dump(coco_results, open('coco_results.json', 'w'))

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

