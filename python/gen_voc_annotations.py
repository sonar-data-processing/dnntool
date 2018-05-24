#!/usr/bin/python
import os
import argparse
import json
import cv2
import drawing
import fnmatch
import numpy as np
import shutil
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
from annotation_utils import *

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def write_voc_annotation(imgfile, filename, class_name, img_folder, img_size, bbox, annotation_folder):
    root = ET.Element("annotation",  verified="yes")
    ET.SubElement(root, "folder").text = img_folder
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "c").text = imgfile

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_size[1])
    ET.SubElement(size, "height").text = str(img_size[0])
    ET.SubElement(size, "depth").text = str(img_size[2])

    ET.SubElement(root, "segmented").text = "0"

    object = ET.SubElement(root, "object")
    ET.SubElement(object, "name").text = class_name
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"

    bndbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(bbox[0])
    ET.SubElement(bndbox, "ymin").text = str(bbox[1])
    ET.SubElement(bndbox, "xmax").text = str(bbox[2])
    ET.SubElement(bndbox, "ymax").text = str(bbox[3])

    base_name = os.path.splitext(filename)[0]
    out_ann = os.path.join(annotation_folder, base_name+".xml")
    print out_ann
    with open(out_ann, "w") as text_file:
        text_file.write(prettify(root))

parser = argparse.ArgumentParser(
    description="Generate Pascal VOC Annotations")

parser.add_argument(
    "image_folder",
    help="Folder containg the image dataset")

parser.add_argument(
    "voc_annotation_folder",
    help="Folder containg the image dataset",
    default="voc")

labels = ["ssiv_bahia", "jequitaia", "balsa"]

def _main_(args):
    image_folder = args.image_folder
    voc_annotation_folder = args.voc_annotation_folder

    if not os.path.exists(voc_annotation_folder):
        os.makedirs(voc_annotation_folder)

    for root, dirs, files in os.walk(image_folder):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        for name in files:
            img_path = os.path.join(root, name)
            _, filename = os.path.split(img_path)

            img = cv2.imread(img_path)
            label_id, gt = get_rbbox_annotation(img_path)
            box = gt[:4]

            write_voc_annotation(
                img_path,
                filename,
                labels[label_id],
                root,
                img.shape,
                box,
                voc_annotation_folder
            )

            cv2.imshow("voc", img)

            if cv2.waitKey(15) & 0xFF == ord('q'):
                exit()

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)
