import os
import math
import numpy as np
import cv2
import annotation_utils as au

def points(im, pts, colors=(255, 0, 0)):
    n = len(pts)
    for i in range(n):
        pt1 = tuple(np.round(pts[i%n]).astype(int))
        pt2 = tuple(np.round(pts[(i+1)%n]).astype(int))
        cv2.line(im, pt1, pt2, colors, 5)

def bbox(im, bbox, color=(255, 0, 0)):
    bbox = bbox.astype(int)
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 5)
def rbox(img, rb, color=(255, 0, 0)):
    points(img, au.rbox2points(rb), color)
    
def boxes(image, boxes, labels, scores):
    image_h, _, _ = image.shape
    for i, box in enumerate(boxes):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image,
                    labels[i] + ' ' + str(scores[i]),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (0,255,0), 2)

    return image

def rboxes(img, rboxes, color=(255, 0, 0)):
    for rb in rboxes:
        rbox(img, rb, color)