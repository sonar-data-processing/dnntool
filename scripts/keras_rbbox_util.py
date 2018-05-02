import os
import math
import fnmatch
import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def scale(pts, f):
    "scale points"
    for p in pts:
        p[0]=p[0]*f[0]
        p[1]=p[1]*f[1]
    return pts

def to_points(bbox, f=(1.0, 1.0)):
    "receive center x, center y, width, height and angle and converts to points"
    cx, cy, w, h, t = bbox
    angle = t*math.pi/180.0
    b = math.cos(angle)*0.5;
    a = math.sin(angle)*0.5;

    pts = []

    pts.append((cx - a*h - b*w, cy + b*h - a*w))
    pts.append((cx + a*h - b*w, cy - b*h - a*w))
    pts.append((2*cx - pts[0][0], 2*cy - pts[0][1]))
    pts.append((2*cx - pts[1][0], 2*cy - pts[1][1]))

    pts = scale(np.array(pts), f)
    w = distance(pts[0], pts[3])
    h = distance(pts[0], pts[1])

    return pts, cx*f[0], cy*f[1], w, h


def load_annotations(filepath, f):
    "load annotations from a specific image"

    path, filename = os.path.split(filepath)
    base_name = os.path.splitext(filename)[0]
    annotation_path = os.path.join(path, base_name+'.txt')

    with open(annotation_path, 'r') as txtfile:
        # read file annotation file
        id, cx, cy, w, h, t = txtfile.readline().rstrip().split(' ')

        # original rotated bounding box
        rbbox_original = np.array([cx, cy, w, h, t], dtype=np.float32)

        # rescale annotations
        pts, cx, cy, w, h = to_points(rbbox_original, f)

        # create rotated bounding box
        rbbox_resized = np.array([cx, cy, w, h, t], dtype=np.float32)

    return id, rbbox_original, rbbox_resized

def scale_factor(src_sz, target_sz):
    factor = min(float(target_sz[0])/float(src_sz[0]), float(target_sz[1])/float(src_sz[1]))
    return factor

def draw_points(im, pts, center):
    "draw points using opencv"
    n = len(pts)
    for i in range(n):
        pt1 = tuple(pts[i%n].astype(int))
        pt2 = tuple(pts[(i+1)%n].astype(int))
        cv2.circle(im, center, 5, (255, 0,0), 5)
        cv2.line(im, pt1, pt2, (255,0,0), 5)

def draw_points_padding(src, pts, center):
    "draw points with padding"
    dst_x, dst_y, dst_w, dst_h = bouding_box(pts)

    src_h = src.shape[0]
    src_w = src.shape[1]

    if (dst_w < src_w):
        dst_w = src_w
    if (dst_h < src_h):
        dst_h = src_h

    dst = np.zeros((dst_h, dst_w, 3), src.dtype)

    x = (dst_w-src_w) / 2
    y = (dst_h-src_h) / 2

    dst[y:y+src_h, x:x+src_w] = src

    for i in range(len(pts)):
        pts[i][0] = pts[i][0]-dst_x
        pts[i][1] = pts[i][1]-dst_y

    cx = center[0]-dst_x
    cy = center[1]-dst_y
    draw_points(dst, pts,center=(cx,cy))
    return dst

def draw_rrbox(src, rrbox):
    pts, cx, cy, _, _ = to_points(rrbox)
    draw_points(src, pts, (int(cx), int(cy)))

def draw_rrbox_padding(src, rrbox):
    pts, cx, cy, _, _ = to_points(rrbox)
    return draw_points_padding(src, pts, (int(cx), int(cy)))



def build_data(id, rrbox):
    w = rrbox[2]
    h = rrbox[3]
    t = rrbox[4]
    return np.array([id, w, h, t], dtype=np.float32)

def preprare_data(im_path, target_size):
    # load image
    im = image.load_img(im_path)

    # convert to array
    src = image.img_to_array(im)

    # calculate scale factor
    factor = scale_factor(im.size, target_size)

    w = int(float(im.size[0]) * factor)
    h = int(float(im.size[1]) * factor)

    # resize image to target size
    resized = cv2.resize(src, (w,h))

    target_size = target_size + (3,)
    dst = np.zeros(target_size, dtype=np.float32)
    dst[0:h, 0:w] = resized

    id, rrbox_original, rrbox_resized = load_annotations(im_path, (factor, factor))

    # src_c = src.copy()/255.0
    # resized_c = resized.copy()/255.0
    # dst_c = dst.copy()/255.0
    #
    # draw_rrbox(src_c, rrbox_original)
    # draw_rrbox(resized_c, rrbox_resized)
    # draw_rrbox(dst_c, rrbox_resized)
    #
    # cv2.imshow("src", src_c)
    # cv2.imshow("dst", dst_c)
    # cv2.imshow("resized", resized_c)
    # cv2.waitKey()

    y = build_data(id, rrbox_resized)
    x = np.expand_dims(dst, axis=0)

    return x, y


def load_data(path, target_size=(224,224)):
    "It receives a path containg the dataset"

    N = 1500
    im_list = []
    label_list = []
    for root, dirs, files in os.walk(path):

        if not ("jequitaia" in root):
            continue

        files = fnmatch.filter(files, '*.png')

        for name in files[:N]:
            im_path = os.path.join(root, name)
            x, label = preprare_data(im_path, target_size)
            label_list.append(label)
            im_list.append(x)

    img_data = np.array(im_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    Y = label_list

    # Shuffle the dataset
    x, y = shuffle(img_data, Y, random_state=2)

    # split train and valid
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=42)

    # split valid and test
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=43)

    y_train = np.reshape(y_train, (len(y_train), 4))
    y_val = np.reshape(y_val, (len(y_val), 4))
    y_test = np.reshape(y_test, (len(y_test), 4))

    return X_train, y_train, X_val, y_val, X_test, y_test

def distance(pt1, pt2):
    "compute the distance between two points"
    dx = pt2[0]-pt1[0]
    dy = pt2[1]-pt1[1]
    return math.sqrt(dx*dx+dy*dy)

def bouding_box(pts):
    pts=np.rollaxis(pts, 1, 0)
    xmin = np.amin(pts[0])
    ymin = np.amin(pts[1])
    xmax = np.amax(pts[0])
    ymax = np.amax(pts[1])
    return int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)


def draw_points_padding(src, pts, center):
    "draw points with padding"
    dst_x, dst_y, dst_w, dst_h = bouding_box(pts)

    src_h = src.shape[0]
    src_w = src.shape[1]

    if (dst_w < src_w):
        dst_w = src_w
    if (dst_h < src_h):
        dst_h = src_h

    dst = np.zeros((dst_h, dst_w, 3), src.dtype)

    x = (dst_w-src_w) / 2
    y = (dst_h-src_h) / 2

    dst[y:y+src_h, x:x+src_w] = src

    for i in range(len(pts)):
        pts[i][0] = pts[i][0]-dst_x
        pts[i][1] = pts[i][1]-dst_y

    cx = center[0]-dst_x
    cy = center[1]-dst_y
    draw_points(dst, pts,center=(cx,cy))
    return dst

def to_annotation(pts):
    "convert points to annotation (cx, cy, width, height)"
