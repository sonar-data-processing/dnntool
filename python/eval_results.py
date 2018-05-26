import os
import math
import argparse
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(
    description="Generate results")

parser.add_argument(
    "csv_filepath",
    help="CSV file contaning the results")

def _load_results(filepath):
    gt = []
    res = []
    iou = []
    score = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gt += [int(row[0])]
            res += [int(row[1])]
            iou += [float(row[2])]
            score += [float(row[3])]
    return gt, res, iou, score

def _average_recall(gt, res, iou):
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
        if gt[i] == res[i] and iou[i] >= 0.85:
            y_true[i] = 1

    ap = average_precision_score(y_true, y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return precision, recall, ap

def plot_precision_recall(gt, res, iou, score):
    precision, recall, ap = _precision_recall(gt, res, iou, score)
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
    plt.figure()
    plt.step(overlap, recall, color='b', alpha=0.2, where='post')
    plt.fill_between(overlap, recall, step='post', alpha=0.2, color='b')
    plt.xlabel('IoU')
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Recall-IoU curve: AR={0:0.5f}'.format(AR))


def _main_(args):
    gt, res, iou, score = _load_results(args.csv_filepath)
    plot_precision_recall(gt, res, iou, score)
    plot_average_recall(gt, res, iou)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

