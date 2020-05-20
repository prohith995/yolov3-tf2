# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:22:49 2020

@author: nxf55806
"""

import numpy as np
from yolov3_tf2.models_simple import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights
import tensorflow as tf

tiny = False
num_classes = 80

path_weights = './data/yolov3.weights'
path_output = './checkpoints/yolov3.tf'
if tiny:
    path_weights = path_weights[:-8] + '-tiny.weights'
    path_output = path_output[:-3] + '-tiny.tf'


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)
yolo.summary()

load_darknet_weights(yolo, path_weights, tiny)
img = np.random.random((1, 320, 320, 3)).astype(np.float32)
output = yolo(img)

yolo.save_weights(path_output)


