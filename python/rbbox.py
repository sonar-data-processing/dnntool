import os
import cv2
import cv2.cv
import imghdr
import fnmatch
import numpy as np
from imgaug import augmenters as iaa
from annotation_utils import *
from keras.utils import Sequence

def prepare_input(src, target_size):
    img_h, img_w, _ = src.shape
    factor = _scale_factor((img_w, img_h), target_size)

    w = int(float(img_w) * factor)
    h = int(float(img_h) * factor)

    # resize image to target size
    resized = cv2.resize(src, (w, h))

    target_size = target_size + (3,)
    x = np.zeros(target_size, dtype=np.float32)
    x[0:h, 0:w] = resized
    return np.expand_dims(x, axis=0)

def transform(rbox, origin, img_size, target_size):
    factor = _scale_factor(img_size, target_size)
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
        print "Total files: ", len(files)
        files = sorted(files)
        for f in files[:N]:
            filepath = os.path.join(root, f)
            item = {}
            item["img"] = filepath
            item["obj"] = _parse_annotation(filepath)
            data += [item]

    np.random.shuffle(data)
    train_valid_split = int(0.8*len(data))
    valid_data = data[train_valid_split:]
    train_data = data[:train_valid_split]
    return train_data, valid_data

def _parse_rbbox_annotation_file(path):
    with open(path, 'r') as file:
        id, cx, cy, w, h, t = file.readline().rstrip().split(' ')
        obj = {}
        obj['id'] = float(id)
        obj['cx'] = float(cx)
        obj['cy'] = float(cy)
        obj['w'] = float(w)
        obj['h'] = float(h)
        obj['t'] = float(t)
    return obj

def _parse_annotation(imgpath):
    ann_path = get_annotation_path(imgpath)
    return _parse_rbbox_annotation_file(ann_path)

def _scale_factor(src_sz, target_sz):
    factor = min(float(target_sz[0])/float(src_sz[0]), float(target_sz[1])/float(src_sz[1]))
    return factor

class BatchGenerator(Sequence):
    def __init__(self, data, target_size=(224, 224), batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

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
        X_batch = np.zeros((r_bound-l_bound, target_w, target_h, 3))
        y_batch = np.zeros((r_bound-l_bound, 4))
        idx = 0
        for item in self.data[l_bound:r_bound]:
            X, y = self.__parse_item(item, self.target_size)
            X_batch[idx] = X
            y_batch[idx] = y
            idx += 1
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.data)

    def __parse_item(self, item, target_size):
        img = cv2.imread(item["img"])
        img = self.aug_pipe.augment_image(img)
        obj = item["obj"]

        img_h, img_w, _ = img.shape
        factor = _scale_factor((img_w, img_h), target_size)

        w = int(float(img_w) * factor)
        h = int(float(img_h) * factor)

        resized = cv2.resize(img, (w, h))

        target_size = target_size + (3,)
        X = np.zeros(target_size, dtype=np.float32)
        X[0:h, 0:w] = resized

        y = np.array([obj["id"], obj["w"], obj["h"], obj["t"]], dtype=np.float32)
        return X, y
