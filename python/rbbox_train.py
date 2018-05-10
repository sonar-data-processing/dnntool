import os
import time
import rbbox
import rbbox_util
import keras
import argparse
import cv2
import cv2.cv
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
    train_data, valid_data = rbbox.load_data(args.dataset_path, N=50000)

    print "Train Size: ", len(train_data)
    print "Valid Size: ", len(valid_data)

    target_size = (320, 320)
    batch_size = 16
    train_generator = rbbox.BatchGenerator(train_data, batch_size=batch_size, target_size=target_size)
    valid_generator = rbbox.BatchGenerator(valid_data, batch_size=batch_size, target_size=target_size)

    image_input = KL.Input(shape=target_size+(3,))
    model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    x = KL.Flatten(name='flatten')(model.output)
    x = KL.Dense(1024, init='normal', activation='relu') (x)
    x = KL.Dense(1024, init='normal', activation='relu')(x)
    out = KL.Dense(4, init='normal', name='out_bboxes_poses')(x)

    model = Model(input=model.input, output=out)

    if os.path.isfile(weights_path):
        model.load_weights(weights_path)

    model.summary()

    optimizer = Adam(lr=0.01, epsilon=1e-08, decay=0.0005)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(weights_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)

    model.fit_generator(
                       generator        = train_generator,
                       steps_per_epoch  = len(train_generator) * 8,
                       epochs           = 1000,
                       verbose          = 1,
                       validation_data  = valid_generator,
                       validation_steps = len(valid_generator),
                       callbacks        = [checkpoint, tensorboard],
                       workers          = 3,
                       max_queue_size   = 8)

if __name__ == '__main__':
    args = parser.parse_args()
    _main(args)
