import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import skimage
from imgaug import augmenters as iaa

# import Mask R-CNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

class LeafSegModalConfig(Config):

    # Give the configuration a recognizable name
    NAME = "leafsegv1"

    # no of gpus available and batch size
    GPU_COUNT = 1
    IMAGES_PER_GPU = 10

    # Number of classes 
    NUM_CLASSES = 1 + 2  # Background + phone,laptop and mobile

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 250
    VALIDATION_STEPS = 70

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # backbone network either resnet50 or resnet101
    BACKBONE = "resnet50"
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    TRAIN_ROIS_PER_IMAGE = 32
    
    MAX_GT_INSTANCES = 20
    
    DETECTION_MAX_INSTANCES = 3

    USE_MINI_MASK = False
    
    #MEAN_PIXEL = np.array([94.8, 143.6, 127.4])

class LeafSegModalConfigTwo(Config):

    # Give the configuration a recognizable name
    NAME = "leafsegv2"

    # no of gpus available and batch size
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5

    # Number of classes 
    NUM_CLASSES = 1 + 1  # Background + phone,laptop and mobile

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 100

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # backbone network either resnet50 or resnet101
    BACKBONE = "resnet50"
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    TRAIN_ROIS_PER_IMAGE = 32
    
    MAX_GT_INSTANCES = 20
    
    DETECTION_MAX_INSTANCES = 10

    USE_MINI_MASK = False
    
    #MEAN_PIXEL = np.array([94.8, 143.6, 127.4])

class InferenceConfig(LeafSegModalConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class InferenceModel2Config(LeafSegModalConfigTwo):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_model_weights(model):

    #model logs path
    MODEL_DIR = os.path.join("D:\IIT\Fyp\Implementation\MRCNN", "logs")
    # print(MODEL_DIR)

    if model == 1 :
        #weight path
        MODEL_PATH = os.path.join("D:\IIT\Fyp\Implementation\MRCNN", "mask_rcnn_leafsegv1_0030.h5")
        # print(MODEL_PATH)

        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=MODEL_DIR)

        # Load trained weights
        print("Loading weights from ", MODEL_PATH)
        model.load_weights(MODEL_PATH, by_name=True)
        model.keras_model._make_predict_function()
        return model
    
    else :
        #weight path
        MODEL_PATH = os.path.join("D:\IIT\Fyp\Implementation\MRCNN", "mask_rcnn_leafsegv2_0052.h5")
        # print(MODEL_PATH)

        inference_config = InferenceModel2Config()

        model= modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

        # Load trained weights
        print("Loading weights from ", MODEL_PATH)
        model.load_weights(MODEL_PATH, by_name=True)
        model.keras_model._make_predict_function()
        return model
    
def get_resized_image(image):
    inference_config = InferenceConfig()

    resized_image, window, scale, padding, crop = utils.resize_image(image,
                                                                 min_dim=inference_config.IMAGE_MIN_DIM,
                                                                 min_scale=inference_config.IMAGE_MIN_SCALE,
                                                                 max_dim=inference_config.IMAGE_MAX_DIM)
    
    return resized_image
   

   

    
