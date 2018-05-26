import sys
import skimage.io as io
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(
    description="Load COCO results format")


parser.add_argument(
    "annotation_filepath",
    help="Folder containg the image dataset")

args = parser.parse_args()

coco=COCO(args.annotation_filepath)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['jequitaia'])
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)

# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
print annIds
anns = coco.loadAnns(annIds)
print anns
coco.showAnns(anns)
plt.show()