import csv
import numpy as np
import annotation_utils
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

LABELS = ["ssiv_bahia", "jequitaia", "balsa"]

def _average_recall(gt, res, iou):
    for i, dt in enumerate(res):
        if gt[i] != dt:
            iou[i] = 0

    all_iou= sorted(iou)
    num_pos = len(all_iou)
    dx = 0.001
  
    overlap = np.arange(0, 1, dx)
    overlap[-1]= 1

    N = len(overlap)
    recall = np.zeros(N, dtype=np.float32)
    for i in range(N):
        recall[i] = (all_iou > overlap[i]).sum() / float(num_pos)

    good_recall = recall[np.where(overlap > 0.5)]
    AR = 2 * dx * np.trapz(good_recall)
    return overlap, recall, AR

def _precision_recall(gt, res, iou, score):
    N = len(gt)
    y_true = np.zeros(N, dtype=int)
    y_score = score

    for i in range(N):
        if gt[i] == res[i] and iou[i] >= 0.5:
            y_true[i] = 1

    ap = average_precision_score(y_true, y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return precision, recall, ap

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

def plot_precision_recall(gt, res, iou, score):
    precision, recall, ap = _precision_recall(gt, res, iou, score)
    print "Average Precision: {}".format(ap)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.5f}'.format(ap))

def plot_average_recall(gt, res, iou):
    overlap, recall, AR = _average_recall(gt, res, iou)
    print "Average Recall: {}".format(AR)
    plt.figure()
    plt.step(overlap, recall, color='b', alpha=0.2, where='post')
    plt.fill_between(overlap, recall, step='post', alpha=0.2, color='b')
    plt.xlabel('IoU')
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Recall-IoU curve: AR={0:0.5f}'.format(AR))

def calc_iou_rbox(shape, gt_rbox, dt_rbox):
    mask_gt = _create_rbbox_mask(shape, gt_rbox)
    mask_dt = _create_rbbox_mask(shape, dt_rbox)
    return _calc_iou(mask_gt, mask_dt)

def load_detection_results(filepath):
    imgs = []
    ids = []
    boxes = []
    rboxes = []
    scores = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            imgs += [row[0]]
            ids += [int(row[1])]
            boxes += [np.array(row[2:6], dtype=np.float32)]
            rboxes += [np.array(row[6:11], dtype=np.float32)]
            scores += [float(row[11])]
    return imgs, ids, boxes, rboxes, scores
