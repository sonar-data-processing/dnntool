import os
import math
import numpy as np
import cv2
import cv2.cv as cv

def points(im, pts):
    n = len(pts)
    for i in range(n):
        pt1 = tuple(np.round(pts[i%n]).astype(int))
        pt2 = tuple(np.round(pts[(i+1)%n]).astype(int))
        cv2.line(im, pt1, pt2, (255,0,0), 5)
def bbox(im, bbox):
    bbox = bbox.astype(int)
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 5)
