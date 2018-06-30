import os
import math
import argparse
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import eval_util
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(
    description="Generate results")

parser.add_argument(
    "csv_filepath",
    help="CSV file contaning the results")

def _load_results(filepath):
    gts = []
    dts = []
    iou = []
    score = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gts += [int(row[0])]
            dts += [int(row[1])]
            iou += [float(row[2])]
            score += [float(row[3])]
    return gts, dts, iou, score

def _main_(args):
    gts, dts, iou, score = _load_results(args.csv_filepath)
    eval_util.plot_precision_recall(gts, dts, iou, score)
    eval_util.plot_average_recall(gts, dts, iou)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)

