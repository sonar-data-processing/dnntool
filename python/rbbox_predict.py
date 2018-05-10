#!/usr/bin/python
import os
import time
import fnmatch
import numpy as np
import rbbox_util as util
import keras
import cv2
import cv2.cv as cv
import keras.layers as KL
from keras.models import Model
from keras.preprocessing import image

def show_results(im, x, preds, target_size):
    factor = util.scale_factor(im.size, target_size)
    cx = im.size[0]/2.0
    cy = im.size[1]/2.0
    w = preds[0][1] / factor
    h = preds[0][2] / factor
    t = preds[0][3]

    mat = image.img_to_array(im)/255.0
    out = util.draw_rrbox_padding(mat, np.array([cx, cy, w, h, t], dtype=np.float32))

    cv2.imshow("result", out)
    cv2.waitKey(200);


image_input = KL.Input(shape=(224, 224, 3))
model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
x = KL.Flatten(name='flatten')(model.output)
x = KL.Dense(1024, init='normal', activation='relu') (x)
x = KL.Dense(1024, init='normal', activation='relu')(x)
out = KL.Dense(4, init='normal', name='out_bboxes_poses')(x)

model = Model(input=model.input, output=out)

model.load_weights("/home/gustavoneves/sources/dnntool/weights/rbbox-weights.hdf5")

# model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

n = 20000
N = 300
im_list = []
for root, dirs, files in os.walk("/home/gustavoneves/data/gemini/dataset/bbox"):

    if not ("jequitaia" in root):
        continue

    files = fnmatch.filter(files, '*.png')
    for name in files[n:n+N]:
        im_path = os.path.join(root, name)

        # load image
        im = image.load_img(im_path)

        x, label = util.preprare_data(im_path, (224,224))
        preds = model.predict(x)

        show_results(im, x, preds, (224,224))


        # # load image
        # im = image.load_img(im_path)
        #
        # # resize image
        # im_r = im.resize((224,224))
        # # convert image to array
        # x = image.img_to_array(im_r)
        # # expand dimension to be (1, w, h, 3)
        # x = np.expand_dims(x, axis=0)
        # # load image ground truth
        # rbbox = util.load_annotations(im_path, im.size, im_r.size)
        #
        # preds = model.predict(x)
        # print "preds: ", preds, "annotation: ", rbbox
        #
        # fx = float(im.size[0]) / float(im_r.size[0])
        # fy = float(im.size[1]) / float(im_r.size[1])
        #
        # pts1, cx1, cy1, w1, h1  = util.to_points(preds[0])
        # canvas = util.draw_points_padding(image.img_to_array(im_r)/255.0, pts1, center=(int(cx1), int(cy1)))
        # cv2.imshow('resized', canvas)
        #
        # pts2, cx2, cy2, w2, h2  = util.to_points(preds[0], (fx, fy))
        # canvas = util.draw_points_padding(image.img_to_array(im)/255.0, pts2, center=(int(cx2), int(cy2)))
        #
        # cv2.imshow('original', canvas)
        # cv2.waitKey()
