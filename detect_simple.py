# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:57:24 2020
@author: nxf55806
"""

# import time
import cv2
# import numpy as np
import tensorflow as tf
from yolov3_tf2.models_simple import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

path_classes = './data/coco.names'
path_weights = './checkpoints/yolov3.tf'
path_weights_tiny = './checkpoints/yolov3-tiny.tf'
resize = 416
num_classes = 80
input_image = './data/girl.png'
tfrecord = None
output_image = './data/girl_out_simple_test.jpg'

tiny = True
if tiny:
    output_image = output_image[:-4] + '_tiny.jpg'


physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
    yolo.load_weights(path_weights_tiny).expect_partial()
else:
    yolo = YoloV3(classes=num_classes)
    yolo.load_weights(path_weights).expect_partial()

class_names = [c.strip() for c in open(path_classes).readlines()]

if tfrecord:
    dataset = load_tfrecord_dataset(
        tfrecord, path_classes, resize)
    dataset = dataset.shuffle(512)
    img_raw, _label = next(iter(dataset.take(1)))
else:
    img_raw = tf.image.decode_image(
        open(input_image, 'rb').read(), channels=3)

img = tf.expand_dims(img_raw, 0)
img = transform_images(img, resize)

# t1 = time.time()
boxes, scores, classes, nums = yolo(img)
# t2 = time.time()
# print('time : {}'.format(t2 - t1))

# for i in range(nums[0]):
#     print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
#                                         np.array(scores[0][i]),
#                                         np.array(boxes[0][i])))

img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
cv2.imwrite(output_image, img)
