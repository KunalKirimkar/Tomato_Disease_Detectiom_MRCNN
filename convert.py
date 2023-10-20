import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
import mrcnn.model as modellib # https://github.com/matterport/Mask_RCNN/
from mrcnn.config import Config
import keras.backend as ke

PATH_TO_SAVE_FROZEN_PB ="./"
FROZEN_NAME ="saved_model.pb"

sess = tf.compat.v1.Session()
with sess.graph.as_default():
    ke.set_session(sess)
    model = tf.keras.models.load_model('D:/Tomato_Disease_Detection_MRCNN/logs/object20231005T1159/mask_rcnn_object_0010.h5')
tf.compat.v1.initialize_all_variables().run(session=sess)

def load_model(Weights):
        global model, graph
        class InferenceConfig(Config):
                NAME = "coco"
                NUM_CLASSES = 1 + 10
                IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES
                DETECTION_MAX_INSTANCES = 100
                DETECTION_MIN_CONFIDENCE = 0.7
                DETECTION_NMS_THRESHOLD = 0.3
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1

        config = InferenceConfig()
        Weights = Weights
        Logs = "D:/Tomato_Disease_Detection_MRCNN/logs/object20230922T1348/"
        model = modellib.MaskRCNN(mode="inference", config=config,
		                  model_dir=Logs)
        # model.load_weights(Weights, by_name=True)
        model.load_weights( Weights, by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "rpn_model"])
        graph = tf.compat.v1.get_default_graph()


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

def freeze_model(model, name):
    frozen_graph = freeze_session(
         sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = PATH_TO_SAVE_FROZEN_PB
    tf.compat.v1.train.write_graph(frozen_graph, directory, name , as_text=False)

def keras_to_tflite(in_weight_file, out_weight_file):
    sess = tf.compat.v1.Session()
    ke.set_session(sess)
    load_model(in_weight_file)
    global model
    freeze_model(model.keras_model, FROZEN_NAME)
    
    # https://github.com/matterport/Mask_RCNN/issues/2020#issuecomment-596449757
    input_arrays = ["input_image"]
    output_arrays = ["mrcnn_class/Softmax","mrcnn_bbox/Reshape"]
    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
        PATH_TO_SAVE_FROZEN_PB+"/"+FROZEN_NAME,
        input_arrays, output_arrays,
        input_shapes={"input_image":[1,256,256,3]}
        )
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open(out_weight_file, "wb").write(tflite_model)
    print("*"*80)
    print("Finished converting keras model to Frozen tflite")
    print('PATH: ', out_weight_file)
    print("*" * 80)

keras_to_tflite("./mask_rcnn_coco.h5","./mask_rcnn_coco.tflite")