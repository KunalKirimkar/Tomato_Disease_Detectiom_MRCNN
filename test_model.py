import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#import custom

# Root directory of the project
ROOT_DIR = "D:/Tomato_Disease_Detection_MRCNN"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


WEIGHTS_PATH = "D:/Tomato_Disease_Detection_MRCNN/logs/object20230922T1348/mask_rcnn_object_0010.h5" 
# change it Using TensorFlow backend.


class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 4

    # 28 classes  # Background + Apple_Black_Rot, Apple_Healthy, Apple_Rust, Apple_Scab, Blueberry_Healthy, Corn_Common_Rust,
    #Corn_Gray_Leaf_Spot,Corn_Healthy, Grape_Black_Measles, Grape_Black_Rot, Grape_Healthy, Peach_Bacterial_Spot, Peach_Healthy,
    #Pepper_Bell_Bacterial_Spot, Pepper_Bell_Healthy, Pepper_Early_Blight, Potato_Healthy, Potato_Late_Blight, Rasberry_Healthy,
    #Soybean_Healthy,Strawberry_Leaf_Scroch,Strawberry_Healthy,Tomato_Bacterial_Spot,Tomato_Early_Blight, Tomato_Healthy,
    #Tomato_Late_Blight,Tomato_Leaf_Spot,Tomato_Target_Spot
    
    NUM_CLASSES = 1 + 10  

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

config = CustomConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "/dataset/")
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 2
    IMAGES_PER_GPU = 4 #GPU with 24GB memory, which can fit 4 images. Adjust down if you use a smaller GPU.
    DETECTION_MIN_CONFIDENCE = 0.9

config = InferenceConfig()
config.display()


# Code for Customdataset class. Same code is present in custom.py file also
class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "Tomato_Bacterial_spot")
        self.add_class("object", 2, "Tomato_Early_blight")
        self.add_class("object", 3, "Tomato_healthy")
        self.add_class("object", 4, "Tomato_Late_blight")
        self.add_class("object", 5, "Tomato_Leaf_Mold")
        self.add_class("object", 6, "Tomato_Septoria_leaf_spot")
        self.add_class("object", 7, "Tomato_Spider_mites_Two_spotted_spider_mite")
        self.add_class("object", 8, "Tomato_Target_Spot")
        self.add_class("object", 9, "Tomato_Tomato_mosaic_virus")
        self.add_class("object", 10, "Tomato_Yellow_Leaf_Curl_Virus")

     
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # We mostly care about the x and y coordinates of each region
        
        #annotations1 = json.load(open('D:/python3.6.8_tensorflow_1.14_env/maskrcnn_leave_disease_detection/dataset/train/train.json'))
        annotations1 = json.load(open(os.path.join(dataset_dir, 'D:/Tomato_Disease_Detection_MRCNN/dataset/val/train.json')))
        #keep the name of the json files in the both train and val folders
        
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys



        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {
                "Tomato_Bacterial_spot": 1,
                "Tomato_Early_blight": 2,
                "Tomato_healthy": 3,
                "Tomato_Late_blight":4,
                "Tomato_Leaf_Mold":5,
                "Tomato_Septoria_leaf_spot":6,
                "Tomato_Spider_mites_Two_spotted_spider_mite":7,
                "Tomato_Target_Spot":8,
                "Tomato_Tomato_mosaic_virus":9,
                "Tomato_Yellow_Leaf_Curl_Virus":10
            }

            #name_dict = {"Horse": 1,"Man": 2} #,"xyz": 3}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]

            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # y, x, width, height = rect
            # rr, cc = skimage.draw.rectangle(y, x, width, height)
            rr = np.clip(rr, 0, info["height"] - 1)
            cc = np.clip(cc, 0, info["width"] - 1)  
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# Inspect the model in training or inference modes values: 'inference' or 'training'
TEST_MODE = "inference"
ROOT_DIR = "D:/Tomato_Disease_Detection_MRCNN/dataset" #if error remove dataset

def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in all visualizations in the notebook.  Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
  _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
  return ax

# Load validation dataset
# Must call before using the dataset
CUSTOM_DIR = "D:/Tomato_Disease_Detection_MRCNN/dataset"
dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

config = CustomConfig()
#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load COCO weights Or, load the last model you trained
weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

#Now, we are ready for testing our model on any image.

#RUN DETECTION
image_id = random.choice(dataset.image_ids)
#image_id = 'D:/MaskRCNN-aar/Dataset/val/1.jfif'
print("image id is :",image_id)
# image, image_meta, gt_class_id, gt_bbox, gt_mask =\
# modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
# info = dataset.image_info[image_id]
# print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,dataset.image_reference(image_id)))

# # Run object detection
# results = model.detect([image], verbose=1)

# # Display results
# x = get_ax(1)
# r = results[0]
# ax = plt.gca()
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

images = []
for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    images.append(image)
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect(images, verbose=1)

for i, image in enumerate(images):
    r = results[i]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'])
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

# This is for predicting images which are not present in dataset
path_to_new_image = 'D:/Tomato_Disease_Detection_MRCNN/test_imgs/Tomato_Target_Spot (457).JPG'
image1 = mpimg.imread(path_to_new_image)

# Run object detection
print(len([image1]))
results1 = model.detect([image1], verbose=1)

# Display results
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
dataset.class_names, r1['scores'], ax=ax, title="Predictions1")

path_to_new_image = 'D:/Tomato_Disease_Detection_MRCNN/test_imgs/Tomato_healthy (368).JPG'
image1 = mpimg.imread(path_to_new_image)

# Run object detection
print(len([image1]))
results1 = model.detect([image1], verbose=1)


# Display results
ax = get_ax(1)
r1 = results1[0]
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
dataset.class_names, r1['scores'], ax=ax, title="Predictions1")