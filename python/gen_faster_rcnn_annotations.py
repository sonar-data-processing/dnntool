#!/usr/bin/python
import os
import cv2
import fnmatch
import argparse
import drawing
import rbbox
import annotation_utils as au
from enum import Enum
from sklearn.utils import shuffle
from annotation_utils import *


class Color(Enum):
    red = 'red'
    blue = 'blue'
    green = 'green'

    def __str__(self):
        return self.value

parser = argparse.ArgumentParser(
    description="Generate faster-rcnn annotations")

parser.add_argument(
    "dataset_folder",
    help="path to a folder with the images and annotations")

parser.add_argument(
    "--output",
    default="annotations.txt",
    help="path to a folder with the images and annotations")

parser.add_argument(
    '--type',
    default='box',
    choices=list(['box', 'rbox']),
    help="Annotation type")

parser.add_argument(
    '-t',
    '--total',
    default=1000,
    type=int,
    help="Total of samples per class")

def _main_(args):
    data_folder = args.dataset_folder
    output_folder = args.output
    ann_type = args.type
    N = args.total
    data = []
    print "Generating faster-rcnn annotations"
    for root, dirs, files in os.walk(args.dataset_folder):
        files = fnmatch.filter(files, '*.png')
        files = shuffle(files)
        for name in files[:N]:
            img_path = os.path.join(root, name)
            _, r = au.get_rbbox_annotation(img_path)
            class_name = os.path.basename(root)
            if (r[8]>=175): r[8] = 0.0

            if ann_type == 'box':
                row = "{},{},{},{},{},{}\n".format(
                    img_path, r[0], r[1], r[2], r[3], class_name)
            else:
                row = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    img_path, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], class_name)
            print row
            data.append(row)

    data = shuffle(data)
    file = open(output_folder, 'w')
    file.writelines(data)
    file.close

    print "Annotations saved  in {}.".format(output_folder) 
 

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)
