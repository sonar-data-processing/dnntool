#!/usr/bin/python
import os
import keras
import imghdr
import numpy as np
import colorsys
import cv2
import random
import cv2.cv as cv
import yad2k.models.keras_yolo as keras_yolo
from PIL import Image, ImageDraw, ImageFont
from keras import backend as K

classes_path = "/home/gustavoneves/sources/dnntool/models/gemini_classes.txt"
# classes_path = "/home/gustavoneves/sources/dnntool/models/coco_classes.txt"
anchors_path = "/home/gustavoneves/sources/dnntool/models/yolo_anchors.txt"
# model_path = "/home/gustavoneves/sources/dnntool/models/yolo.h5"
model_path = "/home/gustavoneves/sources/dnntool/models/gemini.h5"
# test_path = "/home/gustavoneves/sources/dnntool/images"
test_path = ""

def load_classes():
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def load_anchors():
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def prepare_data(src, size):
    resized = cv2.resize(src, size)
    x = np.array(resized, dtype='float32')
    x /= 255.
    x = np.expand_dims(x, 0)
    return x

def run_detection(src, class_names, anchors, size, x, output, sess):
    dst = src.copy()

    h, w, _ = src.shape
    x = prepare_data(src, size)

    feed_dict= {
        model.input: x,
        input_shape: [h, w],
        K.learning_phase(): 0
    }

    out_boxes, out_scores, out_classes = sess.run(output, feed_dict)

    print('Found {} boxes'.format(len(out_boxes)))
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        cv2.rectangle(dst,(left, top),(right, bottom),(0,255,0),3)
    return dst

if __name__ == '__main__':
    print "YOLO Detector"

    class_names = load_classes()
    anchors = load_anchors()

    sess = K.get_session()
    model = keras.models.load_model(model_path)

    model_output_channels = model.layers[-1].output_shape[-1]

    num_classes = len(class_names)
    num_anchors = len(anchors)

    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))
    print "class number: ", num_classes

    model_image_size = model.layers[0].input_shape[1:3]

    yolo_outputs = keras_yolo.yolo_head(model.output, anchors, len(class_names))
    input_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = keras_yolo.yolo_eval(yolo_outputs, input_shape, score_threshold=.3, iou_threshold=.5)

    if test_path == "":
        # cap = cv2.VideoCapture("/home/gustavoneves/Videos/video.mp4")
        cap = cv2.VideoCapture("/home/gustavoneves/data/videos/jequitaia.avi")
        while(True):
            ret, frame = cap.read()
            h, w, _ = frame.shape
            x = prepare_data(frame, model_image_size)
            dst = run_detection(frame, class_names, anchors, model_image_size, x, [boxes, scores, classes], sess)

            cv2.imshow('result', dst)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        for image_file in os.listdir(test_path):
            image_path = os.path.join(test_path, image_file)
            image_type = imghdr.what(image_path)
            if not image_type:
                continue

            src = cv2.imread(image_path)
            x = prepare_data(src, model_image_size)
            dst = run_detection(src, class_names, anchors, model_image_size, x, [boxes, scores, classes], sess)

            cv2.imshow('result', dst)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
    sess.close()


    # for image_file in os.listdir(test_path):
    #     image_path = os.path.join(test_path, image_file)
    #     image_type = imghdr.what(image_path)
    #     if not image_type:
    #         continue
    #
    #     src = cv2.imread(image_path)
    #     # im = src.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    #     resized = cv2.resize(src, model_image_size)
    #     x = np.array(resized, dtype='float32')
    #     x /= 255.
    #     x = np.expand_dims(x, 0)  # Add batch dimension.
    #
    #     h, w, _ = src.shape
    #
    #     feed_dict= {
    #         model.input: x,
    #         input_shape: [h, w],
    #         K.learning_phase(): 0
    #     }
    #
    #     out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict)
    #     print out_boxes
    #     print out_scores
    #     print out_classes


        # font = ImageFont.truetype(
        #     font='font/FiraMono-Medium.otf',
        #     size=np.floor(3e-2 * src.size[1] + 0.5).astype('int32'))
        # thickness = (src.size[0] + src.size[1]) // 300


        # print('Found {} boxes for {}'.format(len(out_boxes), image_file))
        # for i, c in reversed(list(enumerate(out_classes))):
        #     predicted_class = class_names[c]
        #     box = out_boxes[i]
        #     score = out_scores[i]
        #
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #
        #     draw = ImageDraw.Draw(src)
        #
        #     label_size = draw.textsize(label, font)
        #
        #     top, left, bottom, right = box
        #     top = max(0, np.floor(top + 0.5).astype('int32'))
        #     left = max(0, np.floor(left + 0.5).astype('int32'))
        #     bottom = min(src.size[1], np.floor(bottom + 0.5).astype('int32'))
        #     right = min(src.size[0], np.floor(right + 0.5).astype('int32'))
        #     print(label, (left, top), (right, bottom))
        #
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])
        #
        #     print "color: ",colors[c]
        #
        #     # My kingdom for a good redistributable image drawing library.
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        #     draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #     del draw

        # canvas = np.array(src, dtype='float32') / 255.0
        # cv2.imshow("result", canvas)
        # cv2.waitKey()

    sess.close()



    # print "YOLO Detector Python"
    #
    # with open("/home/gustavoneves/sources/dnntool/models/coco_classes.txt") as f:
    #     class_names = f.readlines()
    # class_names = [c.strip() for c in class_names]
    #
    # with open("/home/gustavoneves/sources/dnntool/models/yolo_anchors.txt") as f:
    #     anchors = f.readline()
    #     anchors = [float(x) for x in anchors.split(',')]
    #     anchors = np.array(anchors).reshape(-1, 2)
    #
    # hsv_tuples = [(x / len(class_names), 1., 1.)
    #               for x in range(len(class_names))]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    #
    # random.seed(10101)
    # random.shuffle(colors)
    # random.seed(None)
    # print colors
    #
    # sess = K.get_session()
    # model_path = "/home/gustavoneves/sources/dnntool/models/jequitaia-orientation-source-darknet-13.h5"
    # model1 = keras.models.load_model(model_path)

    # model_output_channels = model.layers[-1].output_shape[-1]
    #
    # num_classes = len(class_names)
    # num_anchors = len(anchors)
    #
    # assert model_output_channels == num_anchors * (num_classes + 5), \
    #     'Mismatch between model and given anchor and class sizes. ' \
    #     'Specify matching anchors and classes with --anchors_path and ' \
    #     '--classes_path flags.'
    # print('{} model, anchors, and classes loaded.'.format(model_path))
    #
    # model_image_size = model.layers[0].input_shape[1:3]

    # yolo_outputs = keras_yolo.yolo_head(model.output, anchors, len(class_names))
    # input_image_shape = K.placeholder(shape=(2, ))
    # boxes, scores, classes = keras_yolo.yolo_eval(yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.5)
    #
