import os
import cv2
import imghdr
import fnmatch
import numpy as np
import keras
import keras.backend as KB
import keras.layers as KL
import csv
from keras.preprocessing import image
from keras.models import Model
from keras.utils import Sequence
from annotation_utils import *


def bbox_rescale(img, target_size):
    img_h, img_w, _ = img.shape
    factor = _scale_factor((img_w, img_h), target_size)

    w = int(float(img_w) * factor)
    h = int(float(img_h) * factor)

    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    target_size = target_size + (3,)
    x = np.zeros(target_size, dtype=np.float32)
    x[0:h, 0:w] = resized
    return x

def prepare_input(src, target_size):
    x = bbox_rescale(src, target_size)
    return np.expand_dims(x, axis=0)

def transform(rbox, origin, img_size, target_size):
    cx = img_size[0]/2.0+origin[0]
    cy = img_size[1]/2.0+origin[1]
    w = rbox[0]
    h = rbox[1]
    t = rbox[2]
    return np.array([cx, cy, w, h, t], dtype=np.float32)

def load_data(path, N=-1):
    data = []
    i = 0
    for root, dirs, files in os.walk(path):
        files = fnmatch.filter(files, '*.png')
        files = sorted(files)
        for f in files[:N]:
            filepath = os.path.join(root, f)
            item = {}
            item["img"] = filepath
            item["obj"] = _parse_annotation(filepath)
            data += [item]


    print "Total data: ", len(data)
    np.random.shuffle(data)
    train_valid_split = int(0.8*len(data))
    valid_data = data[train_valid_split:]
    train_data = data[:train_valid_split]
    return train_data, valid_data

def _parse_obj(data):
    id, x1, y1, x2, y2, cx, cy, w, h, t = data.astype(np.float32)
    obj = {}
    obj['id'] = id
    obj['x1'] = x1
    obj['y1'] = y1
    obj['x2'] = x2
    obj['y2'] = y2
    obj['cx'] = cx
    obj['cy'] = cy
    obj['w'] = w
    obj['h'] = h
    if (abs(t-180)<=5):
        t = t-180
    obj['t'] = t
    return obj


def _parse_rbbox_annotation_file(path):
    with open(path, 'r') as file:
        obj = _parse_obj(np.array(file.readline().rstrip().split(' ')))
    return obj

def _parse_annotation(imgpath):
    ann_path = get_annotation_path(imgpath)
    return _parse_rbbox_annotation_file(ann_path)

def _scale_factor(src_sz, target_sz):
    factor = min(float(target_sz[0])/float(src_sz[0]), float(target_sz[1])/float(src_sz[1]))
    return factor

class BatchGenerator(Sequence):
    def __init__(self, data, target_size=(224, 224), batch_size=32, shuffle=True, use_bbox=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size
        self.use_bbox = use_bbox

        if shuffle: np.random.shuffle(self.data)

    def __len__(self):
        return int(np.ceil(float(len(self.data))/self.batch_size))

    def __getitem__(self, idx):
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.data):
            r_bound = len(self.data)
            l_bound = r_bound-self.batch_size

        target_w, target_h = self.target_size
        x_batch = np.zeros((r_bound-l_bound, target_w, target_h, 3))
        b_batch = np.zeros((r_bound-l_bound, 5))
        y_batch = np.zeros((r_bound-l_bound, 3))
        idx = 0
        for item in self.data[l_bound:r_bound]:
            x, b, y = self.parse_item(item, self.target_size)
            x_batch[idx] = x
            b_batch[idx] = b
            y_batch[idx] = y
            idx += 1
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.data)

    def __parse_bbox(self, obj):
        return np.array([obj['x1'], obj['y1'], obj['x2'], obj['y2']], dtype=np.int32)

    def parse_item(self, item, target_size):
        img = cv2.imread(item['img'])
        img = img / 255.0
        obj = item['obj']
        class_id = int(obj['id'])
        x1, y1, x2, y2 = self.__parse_bbox(obj)
        input_image = None
        if (self.use_bbox):
            subimg = img[y1:y2,x1:x2]
            input_image = bbox_rescale(subimg, target_size)
        else:
            input_image = bbox_rescale(img, target_size)
        
        box = np.array([class_id, x1, y1, x2, y2])
        y = np.array([obj['w'], obj['h'], obj['t']], dtype=np.float32)
        return input_image, box, y

def get_model_rbbox_regressor(target_size=(224, 224)):
    image_shape = target_size+(3,)
    image_input = KL.Input(shape=image_shape)

    resnet50 = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=image_shape)
    box_input = KL.Input(shape=(5,))

    x = KL.Flatten(name='flatten')(resnet50.output)
    x = KL.concatenate([x, box_input])
    x = KL.Dense(1024, init='normal', activation='relu') (x)
    x = KL.Dropout(0.05)(x)
    x = KL.Dense(1024, init='normal', activation='relu')(x)
    x = KL.Dropout(0.05)(x)
    x = KL.Dense(3, init='normal', name='out_bboxes_poses')(x)

    model = Model(inputs=[resnet50.input, box_input], output=x)
    return model

def predict(img, class_id, box, model, target_size=(224, 224), use_bbox=False):
    x1, y1, x2, y2 = np.array(box, dtype=np.int32)
    img = img / 255.0

    if use_bbox:
        subimg = img[y1:y2,x1:x2]
        input_image = prepare_input(subimg, target_size)
    else:
        input_image = prepare_input(img, target_size)

    input_box = np.array([class_id, x1, y1, x2, y2], dtype=np.int32)
    input_box = np.expand_dims(input_box, axis=0)

    preds = model.predict([input_image, input_box])[0]
    w, h = x2-x1, y2-y1

    angle = preds[2]
    if class_id == 1 and h < w/4 and angle >= 5 and angle <= 175:
        preds[2] = 0

    return transform(preds, (x1, y1), (w, h), target_size)

def predict_rbboxes(model, boxes, img, target_size, use_bbox=False):
    img_h, img_w, _ = img.shape
    rboxes=[]
    for box in boxes:
        rbox = predict(
            img, box.get_label(), box.get_rect(img_w, img_h), model,
            target_size=target_size, use_bbox=use_bbox)
        rboxes.append(rbox)
    return np.array(rboxes)

def get_result_filename(img_path, suffix="-result-resnet50"):
    return get_annotation_path(img_path, suffix+".txt")

def save_result(img_path, labels, boxes, rboxes, scores, suffix="-result-resnet50"):
    lines = []
    for idx, l in enumerate(labels):
        b = boxes[idx]
        rb = rboxes[idx]
        line = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
            l,
            b[0], b[1], b[2], b[3],
            rb[0], rb[1], rb[2], rb[3], rb[4],
            scores[idx])
        lines.append(line)
    out_file = get_result_filename(img_path, suffix=suffix)
    file = open(out_file, 'w')
    file.writelines(lines)
    file.close

def load_result(filepath):
    ids = []
    boxes = []
    rboxes = []
    scores = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            ids += [int(row[0])]
            boxes += [np.array(row[1:5], dtype=np.float32)]
            rboxes += [np.array(row[5:10], dtype=np.float32)]
            scores += [float(row[10])]      
    return ids, boxes, rboxes, scores
    
