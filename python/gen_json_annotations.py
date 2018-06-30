#!/usr/bin/python
import os
import cv2
import drawing
import json
import numpy
import annotation_utils as utils
from sklearn.utils import shuffle

def get_gt_mask_filepath(img_path):
    path, filename = os.path.split(img_path)
    base_name = os.path.splitext(filename)[0]
    return os.path.join(path, base_name+'-mask.png')

def load_rbbox(gt):
    rbbox = gt[4:]
    rbbox[0:2] += gt[0:2]
    return rbbox.tolist()

def load_bbox(gt):
    bbox = []
    bbox.append(gt[0])
    bbox.append(gt[1])
    bbox.append(gt[2]-gt[0])
    bbox.append(gt[3]-gt[1])
    return [ int(x) for x in bbox ]

def load_mask(img_path):
    mask = cv2.imread(get_gt_mask_filepath(img_path))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0].flatten().tolist()

def load_rbbox_mask(rbbox):
    mask = utils.rbox2points(rbbox).astype(int)
    box =  cv2.boundingRect(mask)
    return mask.flatten().tolist(), box

def _main_(args):
    import fnmatch
    import random

    base_folder = args.dataset_folder
    if not base_folder[-1] == '/':
        base_folder += '/'

    limit = args.limit
    print limit
    dataset = []
    clsid_cnt = {}
    for root, dirs, files in os.walk(base_folder):
        files = fnmatch.filter(files, '*.png')
        files = fnmatch.filter(files, '*[!-mask].png')
        files = sorted(files)
        for name in files:
            img_path = os.path.join(root, name)
            file_path = img_path.replace(base_folder, '')
            clsid, gt = utils.get_rbbox_annotation(img_path)

            if not limit == None \
                and clsid in clsid_cnt \
                and clsid_cnt[clsid] >= limit:
                break

            img = cv2.imread(img_path)
            img_h, img_w, img_c = img.shape

            gt_rbbox = load_rbbox(gt)
            if args.rbbox_mask:
                gt_mask, gt_bbox = load_rbbox_mask(gt_rbbox)
            else:
                gt_mask = load_mask(img_path)
                gt_bbox = load_bbox(gt)

            item = dict()
            item["filepath"] = file_path
            item["basefolder"] = base_folder
            item["width"] = img_w
            item["height"] = img_h
            item["channels"] = img_c
            item["annotations"] = [] 
            item["annotations"].append({
                "id": clsid,
                "segmentation": gt_mask,
                "bbox": gt_bbox,
                "rbbox": gt_rbbox
                })

            if clsid in clsid_cnt:
                clsid_cnt[clsid] += 1
            else:
                clsid_cnt[clsid] = 1

            dataset += [item]
            print("{} - {}".format(clsid_cnt[clsid], item["filepath"]))

    print dataset

    if args.validation:
        filepath = args.output
        random_dataset = shuffle(dataset)
        path, filename = os.path.split(filepath)
        base_name = os.path.splitext(filename)[0]
        train_filepath = os.path.join(path, "{}_train.json".format(base_name))
        valid_filepath = os.path.join(path, "{}_valid.json".format(base_name))

        n = int(0.8*len(dataset))
        json.dump(random_dataset[:n], open(train_filepath, 'w'))
        json.dump(random_dataset[n:], open(valid_filepath, 'w'))

    json.dump(dataset, open(args.output, 'w'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a json file containing the annotations")

    parser.add_argument(
        "dataset_folder",
        help="The path to a folder with the images and annotations")

    parser.add_argument(
        "--output",
        default="annotations.json",
        help="The output filepath")

    parser.add_argument(
        "--validation",
        default=True,
        help="Save validation dataset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of images per classes")

    parser.add_argument(
        "--rbbox-mask",
        type=bool,
        default=False,
        help="Maximum number of images per classes")

    args = parser.parse_args()

    _main_(args)
