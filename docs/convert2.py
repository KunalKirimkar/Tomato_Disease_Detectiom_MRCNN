import json

import skimage
from mrcnn import utils
from user_config import *
import os
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import mrcnn.model as modellib
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from mrcnn.config import Config
import keras.backend as ke

sess = tf.compat.v1.Session()
ke.set_session(sess)
tf.compat.v1.initialize_all_variables().run(session=sess)

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
            
            
def get_config():
    if is_coco:
        class InferenceConfig():
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUMBER_OF_CLASSES = 1 + 10
            DETECTION_MAX_INSTANCES = 100
            DETECTION_MIN_CONFIDENCE = 0.7
            DETECTION_NMS_THRESHOLD = 0.3

        config = InferenceConfig()

    else:
        config = Config()

    return config

# def load_model(Weights):
#         global model, graph
#         class InferenceConfig(Config):
#                 NAME = "coco"
#                 NUM_CLASSES = 1 + 10
#                 IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES
#                 DETECTION_MAX_INSTANCES = 100
#                 DETECTION_MIN_CONFIDENCE = 0.7
#                 DETECTION_NMS_THRESHOLD = 0.3
#                 GPU_COUNT = 1
#                 IMAGES_PER_GPU = 1

#         config = InferenceConfig()
#         Weights = Weights
#         Logs = "D:/Tomato_Disease_Detection_MRCNN/logs/object20230922T1348/"
#         model = modellib.MaskRCNN(mode="inference", config=config,
# 		                  model_dir=Logs)
#         # model.load_weights(Weights, by_name=True)
#         model.load_weights( Weights, by_name=True,
#         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "rpn_model"])
#         graph = tf.compat.v1.get_default_graph()


# Reference https://github.com/bendangnuksung/mrcnn_serving_ready/blob/master/main.py
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()
        
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def freeze_model(model, name, directory):
    frozen_graph = freeze_session(sess, output_names=[out.op.name for out in model.outputs][:4])
    directory = PATH_TO_SAVE_FROZEN_PB
    tf.compat.v1.train.write_graph(frozen_graph, directory, name , as_text=False)
    print("*"*80)
    print("Finish converting keras model to Frozen PB")
    print('PATH: ', PATH_TO_SAVE_FROZEN_PB)
    print("*" * 80)

def make_serving_ready(model_path, save_serve_path, version_number):
    import tensorflow as tf

    export_dir = os.path.join(save_serve_path, str(version_number))
    graph_pb = model_path

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.compat.v1.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
        tf.compat.v1.import_graph_def(graph_def, name="")
        g = tf.compat.v1.get_default_graph()
        input_image = g.get_tensor_by_name("input_image:0")
        input_image_meta = g.get_tensor_by_name("input_image_meta:0")
        input_anchors = g.get_tensor_by_name("input_anchors:0")

        output_detection = g.get_tensor_by_name("mrcnn_detection/Reshape_1:0")
        output_mask = g.get_tensor_by_name("mrcnn_mask/Reshape_1:0")

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {"input_image": input_image, 'input_image_meta': input_image_meta, 'input_anchors': input_anchors},
                {"mrcnn_detection/Reshape_1": output_detection, 'mrcnn_mask/Reshape_1': output_mask})

        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map=sigs)

    builder.save()
    print("*" * 80)
    print("FINISH CONVERTING FROZEN PB TO SERVING READY")
    print("PATH:", PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL)
    print("*" * 80)

config = get_config()

# def production_ready(model_path:str, frozen_name:str, model_save_dir:str, version_number:int):
#     sess = tf.compat.v1.Session()
#     ke.set_session(sess)
#     config = Config

#     # LOAD MODEL
#     model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)
#     model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
#     # Converting keras model to PB frozen graph
#     freeze_model(model.keras_model, frozen_name, model_save_dir)
    
#     # Now convert frozen graph to Tensorflow Serving Ready
#     make_serving_ready(os.path.join(model_save_dir, frozen_name), model_save_dir, version_number)

#     print("Frozen Model saved here: ", model_save_dir)
#     print("Serving Model saved here: ", model_save_dir)


# model_path = "D:/Tomato_Disease_Detection_MRCNN/logs/object20230922T1348/mask_rcnn_object_0010.h5"
# frozen_name = "export_mrcnn.pb"
# model_save_dir = "D:/Tomato_Disease_Detection_MRCNN/"
# version_number = 1

# production_ready(model_path, frozen_name, model_save_dir, version_number)


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(H5_WEIGHT_PATH, by_name=True)

# Converting keras model to PB frozen graph
freeze_model(model.keras_model, FROZEN_NAME)

# Now convert frozen graph to Tensorflow Serving Ready
make_serving_ready(os.path.join(PATH_TO_SAVE_FROZEN_PB, FROZEN_NAME),
                     PATH_TO_SAVE_TENSORFLOW_SERVING_MODEL,
                     VERSION_NUMBER)

print("COMPLETED")