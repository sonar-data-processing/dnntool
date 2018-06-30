#!/usr/bin/python
import os
import cv2
import drawing
import json
import numpy
import math
import annotation_utils as utils
import tensorflow as tf

def load_dataset(dataset):

    sess = tf.Session()

    for i, item in enumerate(dataset):

        skip = False
        for ann in item["annotations"]:
            if not ann['id'] == 0:
                skip = True
                break
        if skip:
            continue
                
        file_path = item['filepath']
        base_folder = item['basefolder']
        img_path = os.path.join(base_folder, file_path)
        print("{}-{}".format(i, img_path))
        img = cv2.imread(img_path)

        for ann in item["annotations"]:
            bbox = ann["bbox"]
            rbbox = ann["rbbox"]
            segm = ann["segmentation"]
            clsid = ann["id"]

            pts = utils.rbox2points(rbbox)

            width = bbox[2]
            height = bbox[3]

            cx = bbox[0]+width*0.5
            cy = bbox[1]+height*0.5

            # dx0 = (pts[0][0]-bbox[0]) / width
            # dy0 = (pts[0][1]-bbox[1]) / height
            # dx1 = (pts[1][0]-bbox[0]) / width
            # dy1 = (pts[1][1]-bbox[1]) / height
            # dx2 = (pts[2][0]-bbox[0]) / width
            # dy2 = (pts[2][1]-bbox[1]) / height

            dx0 = (pts[0][0]-cx) / width
            dy0 = (pts[0][1]-cy) / height
            dx1 = (pts[1][0]-cx) / width
            dy1 = (pts[1][1]-cy) / height
            dx2 = (pts[2][0]-cx) / width
            dy2 = (pts[2][1]-cy) / height


            x_m = int((pts[1][0]+pts[2][0])/2)
            y_m = int((pts[1][1]+pts[2][1])/2)

            print("dx0: {}, dy0: ".format(dx0, dy0))
            print("dx1: {}, dy1: ".format(dx1, dy1))
            print("dx2: {}, dy2: ".format(dx2, dy2))

            ptx0 = int(cx+dx0*width)
            pty0 = int(cy+dy0*height)
            ptx1 = int(cx+dx1*width)
            pty1 = int(cy+dy1*height)
            ptx2 = int(cx+dx2*width)
            pty2 = int(cy+dy2*height)

            print("rbbox: {}".format(rbbox))

            dd0=pts[1][0]-pts[0][0]
            dd1=pts[1][1]-pts[0][1]

            dd2=pts[2][0]-pts[1][0]
            dd3=pts[2][1]-pts[1][1]

            print("width: {}".format(math.sqrt(dd0 * dd0 + dd1 * dd1)))
            print("height: {}".format(math.sqrt(dd2 * dd2 + dd3 * dd3)))

            x1 = tf.constant(pts[0][0], dtype=tf.float32)
            y1 = tf.constant(pts[0][1], dtype=tf.float32)
            x2 = tf.constant(pts[1][0], dtype=tf.float32)
            y2 = tf.constant(pts[1][1], dtype=tf.float32)
            x3 = tf.constant(pts[2][0], dtype=tf.float32)
            y3 = tf.constant(pts[2][1], dtype=tf.float32)
            rw = tf.sqrt(tf.pow(x2-x1, 2)+tf.pow(y2-y1, 2))
            rh = tf.sqrt(tf.pow(x3-x2, 2)+tf.pow(y3-y2, 2))
            rw = tf.Print(rw, [rw], "Print")
            rh = tf.Print(rh, [rh], "Print")
            print(sess.run(rw))
            print(sess.run(rh))


            pts = utils.rbox2points(rbbox).astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 3)
            segm = numpy.reshape(segm, (-1, 2))
            drawing.points(img, pts, colors=(0, 0, 255))
            cv2.circle(img, (ptx0, pty0) , 5, (255, 0, 0), 3)
            cv2.circle(img, (ptx1, pty1) , 5, (255, 0, 0), 3)
            cv2.circle(img, (ptx2, pty2) , 5, (255, 0, 0), 3)
            cv2.circle(img, (x_m, y_m) , 5, (255, 0, 0), 3)

        cv2.imshow("", img)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            exit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Load a json file with annotations")

    parser.add_argument(
        "json_file",
        help="The json file with annotations")

    args = parser.parse_args()

    with open(args.json_file) as f:
        dataset = json.load(f)
        load_dataset(dataset)