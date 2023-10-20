import os
# User Define parameters

# Make it True if you want to use the provided coco weights
is_coco = False

# keras model file path
H5_WEIGHT_PATH = 'D:/Tomato_Disease_Detection_MRCNN/logs/object20230922T1348/mask_rcnn_object_0010.h5'
MODEL_DIR = os.path.dirname(H5_WEIGHT_PATH)

# Path where the Frozen PB will be save
PATH_TO_SAVE_FROZEN_PB = '/frozen_model/'

# Name for the Frozen PB name
FROZEN_NAME = 'saved_model.pb'

# PATH where to save serving model
PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL = '/serving_model'

# Version of the serving model
VERSION_NUMBER = 1

# Number of classes that you have trained your model
NUMBER_OF_CLASSES = 1 + 10