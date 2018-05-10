#!/usr/bin/python
import os
import argparse
import json
import cv2
import drawing
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
    ET.SubElement(root, "path").text = imgfile

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
    description="Load YOLO Annotations")

parser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    output_folder = config["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    out_anns = os.path.join(output_folder, "annotations")
    if not os.path.exists(out_anns):
        os.makedirs(out_anns)

    out_imgs = os.path.join(output_folder, "images")
    if not os.path.exists(out_imgs):
        os.makedirs(out_imgs)

    anns = load_bbox_annotations(config["annotations"], 3700)
    anns = parse_bbox_annotations(anns)

    for ann in anns:
        _, filename = os.path.split(ann["imgfile"])
        filename = "{}-{}".format(ann["class_name"], filename)
        out_img = os.path.join(out_imgs, filename)
        shutil.copyfile(ann["imgfile"], out_img)
        img = cv2.imread(ann["imgfile"])
        print out_img

        write_voc_annotation(
            out_img,
            filename,
            ann["class_name"],
            "images",
            img.shape,
            ann["bbox"],
            out_anns)

        drawing.bbox(img, ann["bbox"])
        cv2.imshow("img", img)
        if cv2.waitKey(15) & 0xFF == ord('q'):
            exit()

if __name__ == '__main__':
    args = parser.parse_args()
    _main_(args)
