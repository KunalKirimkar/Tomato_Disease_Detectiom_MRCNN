import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = "D:/Tomato_Disease_Detection_MRCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 24GB memory, which can fit 4 images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # Background + Tomato_Bacterial_Spot, Tomato_Early_Blight, Tomato_Healthy, Tomato_Late_Blight, Tomato_Leaf_Mold, Tomato_Septoria_leaf_spot, Tomato_Spider_mites_Two-spotted_spider_mite, Tomato_Target_Spot, Tomato_Tomato_mosaic_virus, Tomato___Tomato_Yellow_Leaf_Curl_Virus

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    


############################################################
#  Dataset
############################################################

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
        
        #annotations1 = json.load(open('D:/Tomato_Disease_Detection_MRCNN/dataset/train/train.json'))
        annotations1 = json.load(open(os.path.join(dataset_dir, 'D:/Tomato_Disease_Detection_MRCNN/dataset/train/train.json')))

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
                # polygons=rectangles,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

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
        
        # mask = np.zeros((5, 5), dtype=np.uint8)
        # start = (1, 1)
        # extent = (3, 3)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # y, x, width, height = rect
            # rr, cc = skimage.draw.rectangle(y, x, width, height)
            rr = np.clip(rr, 0, info["height"] - 1)
            cc = np.clip(cc, 0, info["width"] - 1)  
            mask[rr, cc, i] = 1

        #     rr, cc = skimage.draw.rectangle(start, extent=extent, shape=mask.shape)
        #     mask[rr, cc] = 1

        #     mask
        # np.array([[0, 0, 0, 0, 0],
        #         [0, 1, 1, 1, 0],
        #         [0, 1, 1, 1, 0],
        #         [0, 1, 1, 1, 0],
        #         [0, 0, 0, 0, 0]], dtype=np.uint8)

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
            
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("D:/Tomato_Disease_Detection_MRCNN/dataset", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("D:/Tomato_Disease_Detection_MRCNN/dataset", "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
				
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)	
