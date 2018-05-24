import xml.etree.ElementTree as ET
import os
import json
import fnmatch
import cv2
from annotation_utils import *

labels = ["ssiv_bahia", "jequitaia", "balsa"]

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 20180000000
annotation_id = 0

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, coco_url, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['coco_url'] = coco_url
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox, rbbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(rbbox[0][0])
    seg.append(rbbox[0][1])
    #left_bottom
    seg.append(rbbox[1][0])
    seg.append(rbbox[1][1])
    #right_bottom
    seg.append(rbbox[2][0])
    seg.append(rbbox[2][1])
    #right_top
    seg.append(rbbox[3][0])
    seg.append(rbbox[3][1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def _main(path):

    for root, dirs, files in os.walk(path):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        for name in files:
            
            size = dict()
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            current_image_id = None
            current_category_id = None
            file_name = None
            coco_url = None
            object_name = None

            img_path = os.path.join(root, name)
            img = cv2.imread(img_path)
            label_id, gt = get_rbbox_annotation(img_path)
            _, file_name = os.path.split(img_path)
            coco_url = img_path
            object_name = labels[label_id]
            size['height'], size['width'], size['depth'] = img.shape

            if object_name not in category_set:
                current_category_id = addCatItem(object_name)
            else:
                current_category_id = category_set[object_name]

            current_image_id = addImgItem(file_name, coco_url, size)
            print('add image with {} and {}'.format(file_name, size))

            gt2 = gt.astype(int)
            bbox = []
            bbox.append(gt2[0])
            bbox.append(gt2[1])
            bbox.append(gt2[2]-gt2[0])
            bbox.append(gt2[3]-gt2[1])

            rbbox = gt[4:]
            rbbox[0:2] += bbox[0:2]
            rbbox = rbox2points(rbbox).astype(int)

            print('add annotation with {},{},{},{}.{}'.format(object_name, current_image_id, current_category_id, bbox, rbbox))
            addAnnoItem(object_name, current_image_id, current_category_id, bbox, rbbox)

if __name__ == '__main__':
    xml_path = 'Annotations'
    json_file = 'instances.json'
    _main('/home/gustavoneves/data/gemini/dataset/test/jequitaia/20161206-1642_001166-002416_gemini')
    json.dump(coco, open(json_file, 'w'))