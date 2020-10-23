#%matplotlib inline
import os
from glob import glob
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

from tensorflow.keras.layers import Softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers


import cv2
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

def read_data(directory):
# =============================================================================
#  This function gets a directory name and returns all images in it concatenated
#  to each other
# =============================================================================
    data_list = glob(os.path.join(directory ,r'*.png'))
    #print(os.path.join(directory,'*.png'))
    data = np.asarray([cv2.imread(img,0) for img in data_list])
    return data

dataDir = "E:\Documents\Study\Deep Learning\Project\DATA"
data_name = "train2017"
annFile='{}/annotations/instances_{}.json'.format(dataDir,data_name)


# the index of the image from the set just to visualize
img_idx = 0

# load captions
#coco_caps= COCO(os.path.join(dataDir,r"annotations\captions_" + data_name + ".json"))
# Load insatnces -> here we take the masks
coco_instances = COCO(os.path.join(dataDir,r"annotations\instances_" + data_name + ".json"))
# Load key points
#coco_key_p = COCO(os.path.join(dataDir,r"annotations\person_keypoints_" + data_name + ".json"))

# in the .getCatIds(catNms=[list of all ids we want to filter from the data]);
catIds = coco_instances.getCatIds(catNms=['dog','person','cat']);
# get all the image id for all images based on the filter
imgIds = coco_instances.getImgIds(catIds=catIds);

# Visualize the img_idx specified up in the begining
#
# I = io.imread(os.path.join(dataDir, data_name +r"//" +str(imgIds[img_idx]).zfill(12) + ".jpg"))
# plt.imshow(I);
# plt.axis('off')
# plt.title("image id: " + str(imgIds[img_idx]))
# # and plot the mask
# annIds = coco_instances.getAnnIds(imgIds=imgIds[img_idx], catIds=catIds, iscrowd=None)
# anns = coco_instances.loadAnns(annIds)
# coco_instances.showAnns(anns)

#plot key points
#annIds_keys = coco_key_p.getAnnIds(imgIds=imgIds[idx], catIds=catIds, iscrowd=None)
#anns_keys = coco_key_p.loadAnns(annIds_keys)
#coco_key_p.showAnns(anns_keys)

#print the caption
#annIds = coco_caps.getAnnIds(imgIds=imgIds[idx]);
#anns = coco_caps.loadAnns(annIds)
#coco_caps.showAnns(anns)

#visualize the mask of the annotaion
# mask = int()
# for i in range(len(anns)):
#     anns_img = np.zeros((I.shape[0],I.shape[1]))
#     #mask += coco_instances.annToMask(anns[i])*anns[i]['category_id']
#     mask += np.maximum(anns_img,coco_instances.annToMask(anns[i])*anns[i]['category_id'])
# plt.imshow(mask)
# plt.axis('off')