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
import eval_results

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

def _calc_iou(mask_gt, mask_r):
    mask_i = np.zeros(mask_gt.shape, dtype=np.uint8)
    cv2.bitwise_and(mask_gt, mask_r, mask_i)
    inter_area = np.sum(mask_i)/255.
    gt_area = np.sum(mask_gt)/255.
    r_area = np.sum(mask_r)/255.
    return inter_area / (gt_area + r_area - inter_area)

def gen_results(args):
    gt = []
    res = []
    iou = []
    score = []

    suffix = _get_suffix(args.detector)
    results = []
    interrupt = False

    for root, dirs, files in os.walk(args.image_folder):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        for name in files:
            img_path = os.path.join(root, name)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape

            gt_id, gt_box, gt_rbox = _get_ground_truth(img_path)

            result_path = rbbox.get_result_filename(img_path, suffix='-'+suffix)
            ids, boxes, rboxes, scores = rbbox.load_result(result_path)

            labels = [yolo_labels[id] for id in ids]

            drawing.boxes(img, boxes, labels, scores)
            drawing.rboxes(img, rboxes)
            drawing.rbox(img, gt_rbox, color=(0, 0, 255))

            if rboxes:
                mask_gt = _create_rbbox_mask((img_h, img_w), gt_rbox)
                for i, rbox in enumerate(rboxes):
                    mask_r = _create_rbbox_mask((img_h, img_w), rbox)
                    iou_c = _calc_iou(mask_gt, mask_r)
                    gt += [gt_id]
                    res += [ids[i]]
                    iou += [iou_c]
                    score += [scores[i]]
                    results.append("{}, {}, {}, {}\n".format(
                        gt_id, ids[i], iou_c, scores[i]))
                    print "GT: {}, ID: {}, IoU: {}, Confidence: {}".format(
                        gt_id, ids[i], iou_c, scores[i])
            else:
                gt += [gt_id]
                res += [-1]
                iou += [0]
                score += [0]
                print "GT: {}, ID: {}, IoU: {}, Confidence: {}".format(
                    gt_id, -1, 0, 0)
                results.append("{}, {}, {}, {}\n".format(
                    gt_id, -1, 0, 0))

            cv2.imshow("result", img)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                interrupt = True
                break
        if interrupt:
            break

    filename = suffix+'.csv'
    result_filepath = os.path.join(filename)
    print "Results saved in ", result_filepath
    file = open(result_filepath, 'w')
    file.writelines(results)
    file.close

    return gt, res, iou, score

if __name__ == '__main__':
    args = parser.parse_args()
    gt, res, iou, score = gen_results(args)
    eval_results.plot_precision_recall(gt, res, iou, score)
    eval_results.plot_average_recall(gt, res, iou)