import os
import tensorflow as tf
from custom import CustomConfig
from mrcnn import model as modellib

config = CustomConfig()
ROOT_DIR = "D:/Tomato_Disease_Detection_MRCNN"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs/object20231005T1159/")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Load the model
model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR, config=config)

# Load the weights
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

# def load_detection_model(model):
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     detection_graph = tf.Graph()
#     with detection_graph.as_default():
#         od_graph_def = tf.GraphDef()
#         with tf.gfile.GFile(model, 'rb') as fid:
#             serialized_graph = fid.read()
#             od_graph_def.ParseFromString(serialized_graph)
#             tf.import_graph_def(od_graph_def, name='')
#         input_image = tf.get_default_graph().get_tensor_by_name('input_image:0')
#         input_image_meta = tf.get_default_graph().get_tensor_by_name('input_image_meta:0')
#         input_anchors = tf.get_default_graph().get_tensor_by_name('input_anchors:0')
#         detections = tf.get_default_graph().get_tensor_by_name('mrcnn_detection/Reshape_1:0')
#         mrcnn_mask = tf.get_default_graph().get_tensor_by_name('mrcnn_mask/Sigmoid:0')
#     sessd=tf.Session(config=config,graph=detection_graph)
#     print('Loaded detection model from file "%s"' % model)
#     return sessd, input_image, input_image_meta, input_anchors, detections, mrcnn_mask

# sessd, input_image, input_image_meta, input_anchors, detections, mrcnn_mask = load_detection_model(model_path)
# results = model.detect_pb([image], sessd, input_image, input_image_meta, input_anchors, detections, mrcnn_mask,verbose=1)