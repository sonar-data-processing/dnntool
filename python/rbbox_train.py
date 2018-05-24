#!/usr/bin/python
import os
import time
import rbbox
import keras
import argparse
import cv2
import keras.layers as KL
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

parser = argparse.ArgumentParser(
    description="Load YOLO Annotations")

parser.add_argument(
    "dataset_path",
    help="path to a folder with the images and annotations")

parser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

def _main(args):
    weights_path = args.weights
    train_data, valid_data = rbbox.load_data(args.dataset_path)
    print "Train Size: ", len(train_data)
    print "Valid Size: ", len(valid_data)
    target_size = (224, 224)
    batch_size = 32
    train_generator = rbbox.BatchGenerator(train_data, batch_size=batch_size, target_size=target_size, use_bbox=True)
    valid_generator = rbbox.BatchGenerator(valid_data, batch_size=batch_size, target_size=target_size, use_bbox=True)
    model_rbbox_regr = rbbox.get_model_rbbox_regressor(target_size)
 
    if not weights_path == None and os.path.isfile(weights_path):
        model_rbbox_regr.load_weights(weights_path)

    model_rbbox_regr.summary()

    optimizer = Adam(lr=0.0001, epsilon=1e-08, decay=0.0005)
    model_rbbox_regr.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    scores = model_rbbox_regr.evaluate_generator(
        generator   = valid_generator,
        steps = len(valid_generator),
        use_multiprocessing=True)

    for i in range(len(model_rbbox_regr.metrics_names)):
        print("%s: %.2f" % (model_rbbox_regr.metrics_names[i], scores[i]))

    checkpoint = ModelCheckpoint(weights_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    
    checkpoint.best = scores[0]

    tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    model_rbbox_regr.fit_generator(
                       generator        = train_generator,
                       steps_per_epoch  = len(train_generator) * 8,
                       epochs           = 32,
                       verbose          = 1,
                       validation_data  = valid_generator,
                       validation_steps = len(valid_generator),
                       callbacks        = [checkpoint, tensorboard])

if __name__ == '__main__':
    args = parser.parse_args()
    _main(args)
