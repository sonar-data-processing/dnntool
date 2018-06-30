import os
import math
import argparse
import numpy as np
import cv2
import csv
import drawing
import eval_util
import annotation_utils

parser = argparse.ArgumentParser(
    description="Generate results")

parser.add_argument(
    "detection_result_file",
    help="Detection result file contaning the results")

def _eval_detections(result_file):
    [imgs, dt_ids, dt_boxes, dt_rboxes, scores] = eval_util.load_detection_results(result_file)
    results = []
    print "Evaluating results..."
    for i, img_path in enumerate(imgs):
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        gt_id, gt_box, gt_rbox = annotation_utils.get_ground_truth(img_path)
        iou = eval_util.calc_iou_rbox((img_h, img_w), gt_rbox, dt_rboxes[i])
        results.append("{}, {}, {}, {}\n".format(gt_id, dt_ids[i], iou, scores[i]))

    path, filename = os.path.split(result_file)
    base_name = os.path.splitext(filename)[0]
    result_filepath = os.path.join(base_name + '-eval_results.csv')
    print "Results saved in ", result_filepath
    file = open(result_filepath, 'w')
    file.writelines(results)
    file.close

def _main_(args):
    results = _eval_detections(args.detection_result_file)

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

